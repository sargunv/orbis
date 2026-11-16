//! Render targets (the `dev.sargunv.orbis:targets` preview): the style's
//! layers that draw into targets, read them and draw through them. The
//! style plan, which the tile jobs share, holds their [`Routes`]; the map
//! makes the renderer's [`TargetPlan`] of them.

use orb_render::{CombineMode, ProcessOp, ProcessPlan, TargetPlan};
use orb_style::{CameraInputs, Color, Combine, Gradient, LayerType, Operation, Property, Style};

use crate::style::NodePlan;

/// A target index that draws nowhere and reads as nothing: what a layer
/// naming an undeclared target draws into or through.
const NOWHERE: u32 = u32::MAX;

/// A camera looking straight down, north up.
const LEVEL: CameraInputs = CameraInputs {
    pitch: 0.0,
    bearing: 0.0,
};

/// What the style draws through render targets, by node and target index.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Routes {
    pub(crate) targets: u32,
    /// The target each node draws into, by node ([`NOWHERE`] for one the
    /// style doesn't declare).
    pub(crate) draws: Vec<Option<u32>>,
    /// The target each node (a fill) draws through, by node.
    pub(crate) masks: Vec<Option<u32>>,
    /// The image-process layers that draw, in style order.
    pub(crate) processes: Vec<Route>,
    /// Whether any of their operations reads the camera beyond its zoom.
    pub(crate) camera: bool,
}

/// An image-process layer's place among the nodes and the targets it reads
/// and writes.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Route {
    /// The style layer.
    pub(crate) layer: usize,
    /// It draws before node `at`.
    pub(crate) at: u32,
    pub(crate) source: u32,
    pub(crate) destination: Option<u32>,
    /// Every target it reads: its source and those it combines with.
    reads: Vec<u32>,
}

impl Routes {
    /// The image-process layers, in style order.
    pub(crate) fn layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.processes.iter().map(|p| p.layer)
    }

    /// The renderer's plan, with each process's operations at zoom 0 until
    /// a frame's paint gives them at its own.
    pub(crate) fn render_plan(&self, style: &Style) -> TargetPlan {
        TargetPlan {
            targets: self.targets,
            draws: self.draws.clone(),
            masks: self.masks.clone(),
            processes: (self.processes.iter())
                .map(|p| ProcessPlan {
                    at: p.at,
                    source: p.source,
                    destination: p.destination,
                    ops: ops_at(style, &style.layers[p.layer], 0.0, LEVEL),
                })
                .collect(),
        }
    }
}

/// The style's routes for `nodes`, warning (by layer index) of targets it
/// doesn't declare and of image-process layers that read what they write.
pub(crate) fn plan(
    style: &Style,
    nodes: &[NodePlan],
    warn: &mut impl FnMut(usize, &str),
) -> Routes {
    let layers = &style.layers;
    let uses_targets = layers
        .iter()
        .any(|l| l.destination_target.is_some() || l.source_target.is_some());
    if !uses_targets {
        return Default::default();
    }
    let index = |id: &str| style.targets.iter().position(|t| t.id == id);
    let mut warned = Vec::new();
    let mut resolve = |layer: usize, id: Option<&str>, warn: &mut dyn FnMut(usize, &str)| {
        let id = id?;
        Some(match index(id) {
            Some(i) => i as u32,
            None => {
                if !warned.contains(&layer) {
                    warned.push(layer);
                    warn(
                        layer,
                        &format!("target \"{id}\" not found; the layer draws nothing"),
                    );
                }
                NOWHERE
            }
        })
    };
    let draws = (nodes.iter())
        .map(|n| resolve(n.layer, layers[n.layer].destination_target.as_deref(), warn))
        .collect();
    let masks = (nodes.iter())
        .map(|n| resolve(n.layer, layers[n.layer].source_target.as_deref(), warn))
        .collect();
    let mut processes = Vec::new();
    for (l, layer) in layers.iter().enumerate() {
        if layer.ty != LayerType::ImageProcess {
            continue;
        }
        let source = resolve(l, layer.source_target.as_deref(), warn);
        let destination = resolve(l, layer.destination_target.as_deref(), warn);
        let (Some(source), destination) = (source, destination) else {
            continue;
        };
        if source == NOWHERE || destination == Some(NOWHERE) {
            continue;
        }
        let mut reads = vec![source];
        for op in &layer.operations {
            if let Operation::Combine { target, .. } = op {
                reads.extend(resolve(l, Some(target), warn).filter(|&t| t != NOWHERE));
            }
        }
        processes.push(Route {
            layer: l,
            at: nodes
                .iter()
                .position(|n| n.layer > l)
                .unwrap_or(nodes.len()) as u32,
            source,
            destination,
            reads,
        });
    }
    // A process whose output reaches what it reads draws nothing: of the
    // targets processes write from others, those still left once every
    // target nothing leads into is taken away lie on such a loop.
    let targets = style.targets.len();
    let mut left = vec![true; targets];
    loop {
        let fed = |t: usize| {
            (processes.iter()).any(|p| {
                p.destination == Some(t as u32) && p.reads.iter().any(|&r| left[r as usize])
            })
        };
        let Some(free) = (0..targets).find(|&t| left[t] && !fed(t)) else {
            break;
        };
        left[free] = false;
    }
    processes.retain(|p| {
        let looped = p.destination.is_some_and(|d| left[d as usize])
            && p.reads.iter().any(|&r| left[r as usize]);
        if looped {
            warn(p.layer, "it reads what it draws into; it draws nothing");
        }
        !looped
    });
    let camera = (processes.iter())
        .flat_map(|p| &layers[p.layer].operations)
        .any(Operation::reads_camera);
    Routes {
        targets: targets as u32,
        draws,
        masks,
        processes,
        camera,
    }
}

/// Image-process layer `layer`'s operations at `zoom` and `camera`, for the
/// renderer: targets by index ([`NOWHERE`] for one the style doesn't
/// declare), colours straight.
pub(crate) fn ops_at(
    style: &Style,
    layer: &orb_style::Layer,
    zoom: f64,
    camera: CameraInputs,
) -> Vec<ProcessOp> {
    let target = |id: &str| {
        let at = style.targets.iter().position(|t| t.id == id);
        at.map_or(NOWHERE, |i| i as u32)
    };
    let color = |c: &Property<Color>| c.evaluate_camera(zoom, camera).straight();
    let number = |n: &Property<f32>| n.evaluate_camera(zoom, camera);
    let pair = |p: &Property<[f32; 2]>| p.evaluate_camera(zoom, camera);
    (layer.operations.iter())
        .map(|op| match op {
            Operation::Select { value, tolerance } => {
                let [r, g, b, _] = color(value);
                ProcessOp::Select {
                    color: [r, g, b],
                    tolerance: number(tolerance),
                }
            }
            Operation::Blur { size } => ProcessOp::Blur { size: number(size) },
            Operation::Noise {
                size,
                amount,
                speed,
            } => ProcessOp::Noise {
                size: number(size).max(f32::MIN_POSITIVE),
                amount: number(amount),
                speed: number(speed),
            },
            Operation::Threshold { cutoff } => ProcessOp::Threshold {
                cutoff: number(cutoff),
            },
            Operation::Invert => ProcessOp::Invert,
            Operation::Sharpen { size, amount } => ProcessOp::Sharpen {
                size: number(size),
                amount: number(amount),
            },
            Operation::HighPass { size } => ProcessOp::HighPass { size: number(size) },
            Operation::Convolve { kernel } => ProcessOp::Convolve {
                kernel: kernel.clone(),
            },
            Operation::Offset { translate } => ProcessOp::Offset {
                translate: pair(translate),
            },
            Operation::Dilate { size } => ProcessOp::Dilate { size: number(size) },
            Operation::Erode { size } => ProcessOp::Erode { size: number(size) },
            Operation::Combine { target: id, mode } => ProcessOp::Combine {
                target: target(id),
                mode: match mode {
                    Combine::Multiply => CombineMode::Multiply,
                    Combine::Subtract => CombineMode::Subtract,
                    Combine::Max => CombineMode::Max,
                    Combine::Min => CombineMode::Min,
                    Combine::Over => CombineMode::Over,
                    Combine::Under => CombineMode::Under,
                },
            },
            Operation::Opacity { value } => ProcessOp::Opacity {
                value: number(value),
            },
            Operation::Color { value } => ProcessOp::Color {
                color: color(value),
            },
            Operation::Luminance => ProcessOp::Luminance,
            // A stop before the one ahead of it is held at that one's.
            Operation::Ramp { stops } => ProcessOp::Ramp {
                stops: (stops.iter())
                    .scan(0.0f32, |last, (at, c)| {
                        *last = number(at).clamp(*last, 1.0);
                        Some((*last, color(c)))
                    })
                    .collect(),
            },
            Operation::Split {
                red,
                green,
                blue,
                radial,
            } => ProcessOp::Split {
                channels: [red, green, blue].map(pair),
                radial: *radial,
            },
            Operation::Gradient { kind } => match kind {
                Gradient::Linear { angle } => ProcessOp::Gradient {
                    radial: false,
                    angle: number(angle),
                },
                Gradient::Radial => ProcessOp::Gradient {
                    radial: true,
                    angle: 0.0,
                },
            },
            Operation::Bands { count, width } => ProcessOp::Bands {
                count: number(count),
                width: number(width),
            },
        })
        .collect()
}

/// Whether an image-process layer the style draws shows at `zoom` and
/// `camera` with noise that moves: frames then change with time alone.
pub(crate) fn animates(style: &Style, routes: &Routes, zoom: f64, camera: CameraInputs) -> bool {
    (routes.layers().map(|l| &style.layers[l]))
        .filter(|layer| layer.is_visible_at(zoom))
        .flat_map(|layer| &layer.operations)
        .any(|op| {
            matches!(op, Operation::Noise { speed, .. }
                if speed.evaluate_camera(zoom, camera) > 0.0)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::StylePlan;

    fn plan_of(root: &str, layers: &str) -> StylePlan {
        let previews = orb_style::Previews::NONE.with(orb_style::Preview::Targets);
        let style = orb_style::v8::parse_with(
            &format!(
                r#"{{"version":8,"sources":{{"v":{{"type":"vector","tiles":["x"]}}}},
                    "dev.sargunv.orbis:targets":[{root}],"layers":[{layers}]}}"#
            ),
            previews,
            &[],
        )
        .unwrap();
        StylePlan::of(style)
    }

    #[test]
    fn layers_route_through_targets() {
        let plan = plan_of(
            r#"{"id":"paint"},{"id":"land"}"#,
            r#"{"id":"bg","type":"background"},
               {"id":"water","type":"fill","source":"v","source-layer":"water",
                 "dev.sargunv.orbis:destination-target":"paint"},
               {"id":"mask","type":"dev.sargunv.orbis.image-process",
                 "dev.sargunv.orbis:source-target":"paint","dev.sargunv.orbis:destination-target":"land",
                 "dev.sargunv.orbis:operations":[{"type":"blur","size":2}]},
               {"id":"glow","type":"dev.sargunv.orbis.image-process",
                 "dev.sargunv.orbis:source-target":"land"},
               {"id":"land","type":"fill","dev.sargunv.orbis:source-target":"land"}"#,
        );
        // The background, the water fill and its outline, the land fill
        // (no outline).
        let families: Vec<_> = plan.nodes.iter().map(|n| (n.layer, n.family)).collect();
        use orb_layout::Family::*;
        assert_eq!(
            families,
            [(0, Fill), (1, Fill), (1, FillOutline), (4, Fill)]
        );
        assert_eq!(plan.nodes[3].source, usize::MAX);
        let t = &plan.targets;
        assert_eq!(t.targets, 2);
        assert_eq!(t.draws, [None, Some(0), Some(0), None]);
        assert_eq!(t.masks, [None, None, None, Some(1)]);
        let routes: Vec<_> = (t.processes.iter())
            .map(|p| (p.at, p.source, p.destination))
            .collect();
        assert_eq!(routes, [(3, 0, Some(1)), (3, 1, None)]);
        let ops = &t.render_plan(&plan.style).processes[0].ops;
        assert_eq!(ops, &[ProcessOp::Blur { size: 2.0 }]);
        assert_eq!(t.layers().collect::<Vec<_>>(), [2, 3]);
        assert!(plan.warnings.is_empty(), "{:?}", plan.warnings);
    }

    #[test]
    fn undeclared_targets_and_loops_draw_nothing() {
        let plan = plan_of(
            r#"{"id":"a"},{"id":"b"}"#,
            r#"{"id":"lost","type":"fill","source":"v","source-layer":"x",
                 "dev.sargunv.orbis:destination-target":"nope"},
               {"id":"ab","type":"dev.sargunv.orbis.image-process",
                 "dev.sargunv.orbis:source-target":"a","dev.sargunv.orbis:destination-target":"b"},
               {"id":"ba","type":"dev.sargunv.orbis.image-process",
                 "dev.sargunv.orbis:source-target":"b","dev.sargunv.orbis:destination-target":"a"},
               {"id":"aa","type":"dev.sargunv.orbis.image-process",
                 "dev.sargunv.orbis:source-target":"a"}"#,
        );
        assert_eq!(plan.targets.draws, [Some(NOWHERE), Some(NOWHERE)]);
        // The loop's two draw nothing; what reads it from outside draws.
        assert_eq!(plan.targets.layers().collect::<Vec<_>>(), [3]);
        assert_eq!(plan.warnings.len(), 3, "{:?}", plan.warnings);
    }
}
