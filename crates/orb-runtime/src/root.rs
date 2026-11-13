//! Root transitions: the light, the sky and the terrain ease when an edit
//! changes them, from what the last frame showed to the new style's
//! values. GL JS eases its light and sky by the style's `transition` and
//! each property's `*-transition`; Orbis does too, and also eases the
//! terrain's exaggeration (by the style's `transition`), turning terrain
//! on and off as a rise from flat and a fall back to it, with the fog on
//! it fading alike. No style is absent: an absent `sky` eases as GL JS's
//! transparent one, and is then not drawn.
//!
//! Each eased field starts from the value the frame before the edit
//! showed and eases to the new style's value at the frame's zoom, so once
//! it ends the frame is the new style's, however it got there. A map with
//! nothing easing does no work here beyond evaluating the values.

use orb_render::AtmospherePaint;
use orb_style::{Atmosphere, Light, Style, Transition, TransitionOverride, TranslateAnchor};

use crate::transitions::ease;

/// v8's `light`, evaluated: what the extrusions' shading and the globe's
/// sun read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LightValues {
    pub(crate) anchor: TranslateAnchor,
    /// Premultiplied.
    pub(crate) color: [f32; 4],
    /// Radius, azimuth and polar angle, degrees.
    pub(crate) position: [f32; 3],
    pub(crate) intensity: f32,
}

/// Slots a light eases in.
const LIGHT_SLOTS: usize = 8;

impl LightValues {
    pub(crate) fn of(light: &Light, zoom: f64) -> Self {
        Self {
            anchor: light.anchor.evaluate(zoom),
            color: light.color.evaluate(zoom).premultiplied(),
            position: light.position.evaluate(zoom),
            intensity: light.intensity.evaluate(zoom),
        }
    }

    fn slots(&self) -> [f32; LIGHT_SLOTS] {
        let [r, g, b, a] = self.color;
        let [radius, azimuth, polar] = self.position;
        [r, g, b, a, radius, azimuth, polar, self.intensity]
    }

    fn with(self, s: [f32; LIGHT_SLOTS]) -> Self {
        Self {
            color: [s[0], s[1], s[2], s[3]],
            position: [s[4], s[5], s[6]],
            intensity: s[7],
            ..self
        }
    }
}

/// The light's fields in slot order, with each one's slot count.
const LIGHT: &[(&str, usize)] = &[("color", 4), ("position", 3), ("intensity", 1)];

/// The sky's fields in slot order, with each one's slot count.
const SKY: &[(&str, usize)] = &[
    ("sky-color", 4),
    ("horizon-color", 4),
    ("fog-color", 4),
    ("fog-ground-blend", 1),
    ("horizon-fog-blend", 1),
    ("sky-horizon-blend", 1),
    ("atmosphere-blend", 1),
];

/// The sky's values at `zoom`; GL JS's transparent sky for none.
fn sky_slots(sky: Option<&Atmosphere>, zoom: f64) -> [f32; SKY_SLOTS] {
    let Some(a) = sky else {
        // GL JS's `Sky.setSky(undefined)`: transparent colours, fog-ground
        // blend 1 and no atmosphere; the rest at the spec's defaults.
        let mut s = [0.0; SKY_SLOTS];
        s[12..].copy_from_slice(&[1.0, 0.8, 0.8, 0.0]);
        return s;
    };
    let mut s = [0.0; SKY_SLOTS];
    for (i, c) in [&a.sky_color, &a.horizon_color, &a.fog_color]
        .into_iter()
        .enumerate()
    {
        s[4 * i..4 * i + 4].copy_from_slice(&c.evaluate(zoom).premultiplied());
    }
    for (i, v) in [
        &a.fog_ground_blend,
        &a.horizon_fog_blend,
        &a.sky_horizon_blend,
        &a.atmosphere_blend,
    ]
    .into_iter()
    .enumerate()
    {
        s[12 + i] = v.evaluate(zoom);
    }
    s
}

/// Slots a sky eases in.
const SKY_SLOTS: usize = 16;

fn sky_paint(s: [f32; SKY_SLOTS]) -> AtmospherePaint {
    let color = |i: usize| [s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3]];
    AtmospherePaint {
        sky_color: color(0),
        horizon_color: color(1),
        fog_color: color(2),
        fog_ground_blend: s[12],
        horizon_fog_blend: s[13],
        sky_horizon_blend: s[14],
        atmosphere_blend: s[15],
    }
}

/// The terrain's fields: how far it stands (0 flat or off, 1 on) and its
/// exaggeration while it does.
const TERRAIN: &[(&str, usize)] = &[("presence", 1), ("exaggeration", 1)];

/// What a frame reads from the style's root, eased.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RootFrame {
    pub(crate) light: LightValues,
    pub(crate) sky: Option<AtmospherePaint>,
    /// How far the terrain stands, 0 to 1: its fog fades with it.
    pub(crate) terrain_presence: f64,
    /// The terrain's exaggeration as drawn: the style's, times presence.
    pub(crate) exaggeration: f64,
}

/// The eased root values of a map.
#[derive(Debug, Default)]
pub(crate) struct Root {
    light: Eases<LIGHT_SLOTS>,
    sky: Eases<SKY_SLOTS>,
    terrain: Eases<2>,
}

impl Root {
    /// The style changed from the one the last frame showed to `style`,
    /// with the camera at `zoom`: what differs eases from what was shown.
    pub(crate) fn start(&mut self, style: &Style, zoom: f64) {
        let lookup = |overrides: &[(String, TransitionOverride)], name: &str| {
            (overrides.iter())
                .find(|(n, _)| n == name)
                .map_or(style.transition, |(_, t)| t.over(style.transition))
        };
        let light = LightValues::of(&style.light, zoom).slots();
        self.light
            .start(LIGHT, &light, |name| lookup(&style.light.transitions, name));
        let sky = sky_slots(style.atmosphere.as_ref(), zoom);
        let none = Vec::new();
        let sky_transitions = style.atmosphere.as_ref().map_or(&none, |a| &a.transitions);
        self.sky
            .start(SKY, &sky, |name| lookup(sky_transitions, name));
        let terrain = self.terrain_target(style);
        // Terrain rising from flat rises at its new exaggeration.
        if let Some(shown) = &mut self.terrain.shown
            && shown[0] == 0.0
        {
            shown[1] = terrain[1];
        }
        self.terrain.start(TERRAIN, &terrain, |_| style.transition);
    }

    fn terrain_target(&self, style: &Style) -> [f32; 2] {
        match &style.terrain {
            Some(t) => [1.0, t.exaggeration as f32],
            // Off, it falls at the exaggeration it stood at.
            None => [0.0, self.terrain.shown.map_or(0.0, |s| s[1])],
        }
    }

    /// This frame's values of `style` at `zoom` and `now`, eased.
    pub(crate) fn frame(&mut self, style: Option<&Style>, zoom: f64, now: f64) -> RootFrame {
        let default_light;
        let light = match style {
            Some(s) => &s.light,
            None => {
                default_light = orb_style::v8::default_light();
                &default_light
            }
        };
        let light = LightValues::of(light, zoom);
        let light = light.with(self.light.frame(light.slots(), now));
        let atmosphere = style.and_then(|s| s.atmosphere.as_ref());
        let sky = self.sky.frame(sky_slots(atmosphere, zoom), now);
        let sky = (atmosphere.is_some() || self.sky.running()).then(|| sky_paint(sky));
        let terrain = match style {
            Some(style) => self.terrain_target(style),
            None => [0.0; 2],
        };
        let [presence, exaggeration] = self.terrain.frame(terrain, now).map(f64::from);
        RootFrame {
            light,
            sky,
            terrain_presence: presence,
            exaggeration: presence * exaggeration,
        }
    }

    /// Whether anything eases.
    pub(crate) fn running(&self) -> bool {
        self.light.running() || self.sky.running() || self.terrain.running()
    }

    /// Holds the terrain's rise at its start this frame: a terrain turned
    /// on that waits for its pipelines rises once it draws, rather than
    /// while nothing shows it.
    pub(crate) fn hold_terrain(&mut self) {
        for ease in &mut self.terrain.running {
            ease.span = None;
        }
    }
}

/// Values that ease slot by slot, each field's slots together.
#[derive(Debug)]
struct Eases<const N: usize> {
    /// What the last frame showed.
    shown: Option<[f32; N]>,
    running: Vec<Ease>,
}

impl<const N: usize> Default for Eases<N> {
    fn default() -> Self {
        Self {
            shown: None,
            running: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct Ease {
    slot: usize,
    from: f32,
    /// When it eases, in frame time; `None` until the first frame after
    /// the edit, when its delay starts.
    span: Option<(f64, f64)>,
    transition: Transition,
}

impl<const N: usize> Eases<N> {
    /// The values became `target`: each field of `fields` (names and slot
    /// counts) that differs from what was shown eases by `transition`.
    fn start(
        &mut self,
        fields: &[(&str, usize)],
        target: &[f32; N],
        transition: impl Fn(&str) -> Transition,
    ) {
        let Some(shown) = self.shown else {
            return;
        };
        let mut at = 0;
        for &(name, n) in fields {
            let slots = at..at + n;
            at += n;
            if slots.clone().all(|i| shown[i] == target[i]) {
                continue;
            }
            self.running.retain(|e| !slots.contains(&e.slot));
            let t = transition(name);
            if t.duration + t.delay <= 0.0 {
                continue;
            }
            self.running.extend(slots.map(|slot| Ease {
                slot,
                from: shown[slot],
                span: None,
                transition: t,
            }));
        }
    }

    /// `target` eased at `now`; eases that ended stop.
    fn frame(&mut self, mut target: [f32; N], now: f64) -> [f32; N] {
        self.running.retain_mut(|e| {
            let t = e.transition;
            let (begin, end) = *e
                .span
                .get_or_insert((now + t.delay, now + t.delay + t.duration));
            if now >= end {
                return false;
            }
            let x = ease((now - begin) / (end - begin)) as f32;
            target[e.slot] = e.from + (target[e.slot] - e.from) * x;
            true
        });
        self.shown = Some(target);
        target
    }

    fn running(&self) -> bool {
        !self.running.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn style(json: &str) -> Style {
        orb_style::v8::parse(json).unwrap()
    }

    #[test]
    fn the_sky_eases_by_its_own_transition_and_ends_at_the_new_value() {
        let a = style(r##"{"version":8,"sky":{"sky-color":"#000000"},"sources":{},"layers":[]}"##);
        let b = style(
            r##"{"version":8,"transition":{"duration":1000},
            "sky":{"sky-color":"#ffffff","sky-color-transition":{"duration":100},
            "fog-color":"#000000"},"sources":{},"layers":[]}"##,
        );
        let mut root = Root::default();
        root.frame(Some(&a), 0.0, 0.0);
        root.start(&b, 0.0);
        let first = root.frame(Some(&b), 0.0, 10.0).sky.unwrap();
        assert_eq!(first.sky_color[0], 0.0, "it starts on the next frame");
        let half = root.frame(Some(&b), 0.0, 60.0).sky.unwrap();
        assert!((half.sky_color[0] - 0.5).abs() < 1e-6);
        // The fog takes the style's second.
        assert!(half.fog_color[0] > 0.9);
        let sky = root.frame(Some(&b), 0.0, 120.0).sky.unwrap();
        assert_eq!(sky.sky_color[0], 1.0);
        assert!(root.running());
        let end = root.frame(Some(&b), 0.0, 2000.0).sky.unwrap();
        assert_eq!(end, sky_paint(sky_slots(b.atmosphere.as_ref(), 0.0)));
        assert!(!root.running());
    }

    #[test]
    fn a_removed_sky_fades_to_transparent_then_stops_drawing() {
        let a = style(r##"{"version":8,"sky":{},"sources":{},"layers":[]}"##);
        let b = style(r##"{"version":8,"sources":{},"layers":[]}"##);
        let mut root = Root::default();
        assert!(root.frame(Some(&a), 0.0, 0.0).sky.is_some());
        root.start(&b, 0.0);
        root.frame(Some(&b), 0.0, 1.0);
        let half = root.frame(Some(&b), 0.0, 151.0).sky.unwrap();
        assert!(half.sky_color[3] > 0.0 && half.sky_color[3] < 1.0);
        assert!(root.frame(Some(&b), 0.0, 400.0).sky.is_none());
    }

    #[test]
    fn terrain_rises_from_flat_and_falls_back_at_its_exaggeration() {
        let json = |terrain: &str| {
            format!(
                r#"{{"version":8,{terrain}"sources":{{"dem":{{"type":"raster-dem","tiles":["x/{{z}}/{{x}}/{{y}}"]}}}},"layers":[]}}"#
            )
        };
        let off = style(&json(""));
        let on = style(&json(r#""terrain":{"source":"dem","exaggeration":2},"#));
        let mut root = Root::default();
        assert_eq!(root.frame(Some(&off), 0.0, 0.0).exaggeration, 0.0);
        root.start(&on, 0.0);
        root.frame(Some(&on), 0.0, 0.0);
        let half = root.frame(Some(&on), 0.0, 150.0);
        assert!((half.exaggeration - 1.0).abs() < 1e-6, "{half:?}");
        assert!((half.terrain_presence - 0.5).abs() < 1e-6);
        assert_eq!(root.frame(Some(&on), 0.0, 300.0).exaggeration, 2.0);
        root.start(&off, 0.0);
        root.frame(Some(&off), 0.0, 300.0);
        let half = root.frame(Some(&off), 0.0, 450.0);
        assert!((half.exaggeration - 1.0).abs() < 1e-6, "{half:?}");
        assert_eq!(root.frame(Some(&off), 0.0, 600.0).exaggeration, 0.0);
    }

    #[test]
    fn an_unchanged_root_starts_nothing() {
        let a = style(r##"{"version":8,"sky":{},"sources":{},"layers":[]}"##);
        let mut root = Root::default();
        root.frame(Some(&a), 3.0, 0.0);
        root.start(&a, 3.0);
        assert!(!root.running());
    }
}
