//! v8's `light`, parsed into the style's [`Light`].
use crate::v8::{Cx, layer::typed_property_at};
use crate::*;
use orb_expr::{PropertyType, Type, json::Json};

/// v8's default `light`: every property at its spec default.
pub fn default_light() -> Light {
    parse(&mut Cx::default(), None)
}

/// The light's spec properties that transition (`anchor` doesn't).
const TRANSITIONED: &[&str] = &["position", "color", "intensity"];

/// style-spec#1797's shadows, as maplibre/maplibre#555 names them.
const SHADOW_INTENSITY: &str = "shadow-intensity";
const GROUND_LIGHTING: &str = "dev.sargunv.orbis:ground-lighting";
const AMBIENT_COLOR: &str = "dev.sargunv.orbis:ambient-color";
const AMBIENT_INTENSITY: &str = "dev.sargunv.orbis:ambient-intensity";
const TWILIGHT: &str = "dev.sargunv.orbis:twilight";

/// The light's properties beyond the spec, each behind its preview; all
/// transition.
const PREVIEWS: [(&str, Preview); 5] = [
    (SHADOW_INTENSITY, Preview::Shadows),
    (GROUND_LIGHTING, Preview::GroundLighting),
    (AMBIENT_COLOR, Preview::AmbientColor),
    (AMBIENT_INTENSITY, Preview::AmbientIntensity),
    (TWILIGHT, Preview::Twilight),
];

pub(crate) fn parse(cx: &mut Cx, json: Option<&Json>) -> Light {
    let prop = |name, ty, default, interpolated, minimum, maximum| Prop {
        name,
        ty,
        default: Some(default),
        zoom: true,
        feature: false,
        interpolated,
        minimum,
        maximum,
        tokens: false,
        ..Prop::ROOT
    };
    let number = |name, default, maximum| {
        prop(
            name,
            PropertyType::Number,
            default,
            true,
            Some(0.0),
            Some(maximum),
        )
    };
    let anchor = prop(
        "anchor",
        PropertyType::Enum(&["map", "viewport"]),
        "\"viewport\"",
        false,
        None,
        None,
    );
    let position = prop(
        "position",
        PropertyType::Array(Type::Number, Some(3)),
        "[1.15,210,30]",
        true,
        None,
        None,
    );
    let color = prop("color", PropertyType::Color, "\"white\"", true, None, None);
    // Each property's preview, on when the style uses it with its preview on.
    let mut on = [false; PREVIEWS.len()];
    if let Some(json) = json {
        for member in cx.object(json, "light").unwrap_or_default() {
            let name = member.key.strip_suffix("-transition");
            let base = name.unwrap_or(&member.key);
            if let Some(i) = PREVIEWS.iter().position(|&(n, _)| n == base) {
                let what = format!("\"light.{}\"", member.key);
                let path = format!("light.{}", member.key);
                on[i] |= cx.preview(PREVIEWS[i].1, path, member.key_span, what);
                continue;
            }
            let known = match name {
                Some(name) => TRANSITIONED.contains(&name),
                None => ["anchor", "position", "color", "intensity"].contains(&member.key.as_str()),
            };
            if !known {
                cx.warn_unknown("light", member);
            }
        }
    }
    let mut transitioned = TRANSITIONED.to_vec();
    transitioned.extend((PREVIEWS.iter().zip(on)).filter_map(|(&(n, _), on)| on.then_some(n)));
    // A property beyond the spec reads the style only with its preview on.
    fn read<T: crate::PropertyValue>(
        cx: &mut Cx,
        json: Option<&Json>,
        on: &[bool],
        prop: &Prop,
    ) -> crate::Property<T> {
        let enabled = (PREVIEWS.iter())
            .position(|&(n, _)| n == prop.name)
            .is_none_or(|i| on[i]);
        let value = json.and_then(|j| j.get(prop.name)).filter(|_| enabled);
        typed_property_at(cx, value, &format!("light.{}", prop.name), prop)
    }
    Light {
        anchor: read(cx, json, &on, &anchor),
        position: read(cx, json, &on, &position),
        color: read(cx, json, &on, &color),
        intensity: read(cx, json, &on, &number("intensity", "0.5", 1.0)),
        shadow_intensity: read(cx, json, &on, &number(SHADOW_INTENSITY, "0", 1.0)),
        ground_lighting: read(cx, json, &on, &number(GROUND_LIGHTING, "0", 1.0)),
        ambient_color: read(
            cx,
            json,
            &on,
            &prop(
                AMBIENT_COLOR,
                PropertyType::Color,
                "\"white\"",
                true,
                None,
                None,
            ),
        ),
        ambient_intensity: read(cx, json, &on, &number(AMBIENT_INTENSITY, "0.5", 1.0)),
        twilight: read(cx, json, &on, &number(TWILIGHT, "12", 90.0)),
        transitions: crate::v8::paint::root_transitions(cx, json, "light", &transitioned),
    }
}

#[cfg(test)]
mod tests {
    use crate::{Preview, Previews};

    const DOC: &str = r##"{"version": 8, "sources": {}, "layers": [],
        "light": {"dev.sargunv.orbis:ground-lighting": 1,
                  "dev.sargunv.orbis:ground-lighting-transition": {"duration": 0},
                  "dev.sargunv.orbis:ambient-color": "#203050",
                  "dev.sargunv.orbis:ambient-intensity": ["interpolate", ["linear"], ["zoom"], 0, 0.2, 10, 0.6],
                  "dev.sargunv.orbis:twilight": 20}}"##;

    #[test]
    fn ground_lighting_reads_each_key_behind_its_preview() {
        let on = crate::v8::parse_with(DOC, Previews::ALL, &[]).unwrap();
        assert!(on.warnings.is_empty(), "{:?}", on.warnings);
        let light = &on.light;
        assert_eq!(light.ground_lighting.evaluate(0.0), 1.0);
        assert_eq!(light.ambient_intensity.evaluate(5.0), 0.4);
        assert_eq!(light.twilight.evaluate(0.0), 20.0);
        assert_eq!(
            light.ambient_color.evaluate(0.0).straight()[2],
            0x50 as f32 / 255.0
        );
        let names: Vec<&str> = (light.transitions.iter())
            .map(|(n, _)| n.as_str())
            .collect();
        assert_eq!(names, ["dev.sargunv.orbis:ground-lighting"]);

        // One preview on gates only its own key.
        let only = Previews::NONE.with(Preview::GroundLighting);
        let some = crate::v8::parse_with(DOC, only, &[]).unwrap();
        assert_eq!(some.warnings.len(), 3, "{:?}", some.warnings);
        assert_eq!(some.light.ground_lighting.evaluate(0.0), 1.0);
        assert_eq!(some.light.twilight.evaluate(0.0), 12.0);
        assert_eq!(some.light.ambient_intensity.evaluate(0.0), 0.5);

        let off = crate::v8::parse(DOC).unwrap();
        assert_eq!(off.warnings.len(), 5, "{:?}", off.warnings);
        for (w, preview) in off.warnings.iter().zip([
            Preview::GroundLighting,
            Preview::GroundLighting,
            Preview::AmbientColor,
            Preview::AmbientIntensity,
            Preview::Twilight,
        ]) {
            assert!(w.message.contains(preview.name()), "{}", w.message);
        }
        assert_eq!(off.light.ground_lighting.evaluate(0.0), 0.0);
        assert!(off.light.transitions.is_empty());
    }
}
