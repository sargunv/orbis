//! WebGL 2's stages as the build wrote them in GLSL ES 3.00 (build.rs's
//! `glsl`), each put together from its lines with its constants' values
//! and its depth's range written in.

use std::sync::OnceLock;

use crate::{GLSL_MINUS_ONE_TO_ONE, GLSL_ZERO_TO_ONE};

/// A stage: its lines' numbers in `glsl.ids`, and each override's
/// constant by name and line.
struct Stage {
    entry: &'static str,
    lines: std::ops::Range<usize>,
    constants: &'static [(&'static str, usize)],
}

include!(concat!(env!("OUT_DIR"), "/glsl.rs"));

/// The stages' lines, each once.
const TEXT: &str = include_str!(concat!(env!("OUT_DIR"), "/glsl.txt"));
/// Each stage's lines' numbers in [`TEXT`], `u16`s.
const IDS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/glsl.ids"));

fn lines() -> &'static [&'static str] {
    static LINES: OnceLock<Vec<&str>> = OnceLock::new();
    LINES.get_or_init(|| TEXT.lines().collect())
}

/// Shader `label`'s stage at `entry` as GLSL ES 3.00, for wgpu's GL back
/// end to take as it is: `constants` (override values by name, as a
/// pipeline sets them) written in, and its depth clipped from 0 to 1 where
/// the device does (`zero_to_one_depth`, `EXT_clip_control`), remapped
/// into GL's -1 to 1 elsewhere, as wgpu writes it. `None` for a stage
/// WebGL 2 has none of, or a constant it doesn't take or can't hold.
pub fn glsl(
    label: &str,
    entry: &str,
    constants: &[(&str, f64)],
    zero_to_one_depth: bool,
) -> Option<String> {
    let (_, stages) = STAGES.iter().find(|(l, _)| *l == label)?;
    let stage = stages.iter().find(|s| s.entry == entry)?;
    let taken = |name: &str| stage.constants.iter().any(|&(c, _)| c == name);
    if !constants.iter().all(|&(name, _)| taken(name)) {
        return None;
    }
    let lines = lines();
    let ids = &IDS[stage.lines.start * 2..stage.lines.end * 2];
    let mut out = String::with_capacity(ids.len() * 32);
    for (n, &id) in ids.as_chunks::<2>().0.iter().enumerate() {
        let line = lines[usize::from(u16::from_le_bytes(id))];
        let value = (stage.constants.iter().find(|&&(_, at)| at == n))
            .and_then(|&(name, _)| constants.iter().find(|&&(c, _)| c == name));
        if let Some(&(_, value)) = value {
            // `const <type> <name> = <default>;`
            let (declaration, _) = line.split_once(" = ")?;
            let ty = declaration.split_ascii_whitespace().nth(1)?;
            out.push_str(declaration);
            out.push_str(" = ");
            out.push_str(&literal(ty, value)?);
            out.push(';');
        } else if !zero_to_one_depth && line == GLSL_ZERO_TO_ONE {
            out.push_str(GLSL_MINUS_ONE_TO_ONE);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    Some(out)
}

/// `value` as a GLSL literal of type `ty`, converted as WebGPU converts a
/// pipeline's constants.
fn literal(ty: &str, value: f64) -> Option<String> {
    let integral = |min: f64, max: f64| value.fract() == 0.0 && (min..=max).contains(&value);
    Some(match ty {
        "bool" => (value != 0.0).to_string(),
        "uint" if integral(0.0, f64::from(u32::MAX)) => format!("{}u", value as u32),
        "int" if integral(f64::from(i32::MIN), f64::from(i32::MAX)) => format!("{}", value as i32),
        "float" if (value as f32).is_finite() => format!("{:?}", value as f32),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every stage the build wrote names its constants' lines, and writes
    /// them in.
    #[test]
    fn constants_are_written_in() {
        for (label, stages) in STAGES {
            for stage in *stages {
                let defaults = glsl(label, stage.entry, &[], true).expect("a stage");
                for &(name, _) in stage.constants {
                    let set = [0.0, 1.0].map(|v| glsl(label, stage.entry, &[(name, v)], true));
                    assert!(
                        set.iter().any(|s| s.as_ref() != Some(&defaults)),
                        "{label} {}: `{name}`",
                        stage.entry
                    );
                }
                assert_eq!(glsl(label, stage.entry, &[("nothing", 1.0)], true), None);
            }
        }
        assert!(
            (STAGES.iter().flat_map(|(_, s)| *s))
                .any(|s| s.constants.iter().any(|&(c, _)| c == crate::PLANE)),
            "a stage with a plane variant"
        );
    }

    #[test]
    fn literals() {
        assert_eq!(literal("bool", 0.0).as_deref(), Some("false"));
        assert_eq!(literal("uint", 11.0).as_deref(), Some("11u"));
        assert_eq!(literal("uint", -1.0), None);
        assert_eq!(literal("uint", 0.5), None);
        assert_eq!(literal("int", -3.0).as_deref(), Some("-3"));
        assert_eq!(literal("float", 0.25).as_deref(), Some("0.25"));
    }
}
