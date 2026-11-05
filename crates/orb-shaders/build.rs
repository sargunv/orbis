//! With the `glsl` feature, translates every Compat composition to GLSL ES
//! 3.00 (plan §8.4): one program per shader, with the
//! uniforms naga named for each WGSL binding.

fn main() {
    println!("cargo::rerun-if-changed=src/catalog.rs");
    println!("cargo::rerun-if-changed=src/wgsl");
    #[cfg(feature = "glsl")]
    glsl::write();
}

#[cfg(feature = "glsl")]
#[allow(dead_code, unreachable_pub)]
#[path = "src/catalog.rs"]
mod catalog;

#[cfg(feature = "glsl")]
#[path = "src/translate.rs"]
mod translate;

#[cfg(feature = "glsl")]
mod glsl {
    use std::fmt::Write;

    use crate::catalog::COMPAT_SHADERS;

    pub(crate) fn write() {
        let mut out = String::from(
            "/// (shader label, program)\n\
             pub(crate) static PROGRAMS: &[(&str, orb_hal::GlslProgram<'static>)] = &[\n",
        );
        for shader in COMPAT_SHADERS {
            let source = shader.compat_source();
            let program =
                crate::translate::glsl(shader.label, &source, shader.vertex, shader.fragment, &[])
                    .unwrap_or_else(|e| panic!("{e}"));
            writeln!(
                out,
                "({:?}, orb_hal::GlslProgram {{ vertex: {:?}, fragment: {:?}, blocks: &{:?}, textures: &[{}] }}),",
                shader.label,
                program.vertex,
                program.fragment,
                program.blocks,
                program
                    .textures
                    .iter()
                    .map(|(name, texture, sampler)| format!(
                        "orb_hal::GlslTexture {{ name: {name:?}, texture: {texture:?}, sampler: {sampler:?} }}"
                    ))
                    .collect::<Vec<_>>()
                    .join(", "),
            )
            .expect("writes to a string");
        }
        out.push_str("];\n");
        let dir = std::env::var("OUT_DIR").expect("cargo sets OUT_DIR");
        std::fs::write(format!("{dir}/glsl.rs"), out).expect("writes OUT_DIR");
    }
}
