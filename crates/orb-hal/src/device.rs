//! Known driver bugs: the quirks table, and the compile watchdog's limit.
//! Data and pure logic: hosts bring the adapter's facts.

/// PCI vendor ids, as Vulkan, Metal and WebGPU report them.
pub mod vendor {
    pub const QUALCOMM: u32 = 0x5143;
}

/// What a host knows about a GPU when matching quirks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Adapter<'a> {
    pub api: Api,
    pub vendor: u32,
    pub name: &'a str,
    pub driver: &'a str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    Vulkan,
    Metal,
    Dx12,
    WebGpu,
    Gl,
}

/// How Orbis meets a known driver bug.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Workaround {
    /// Skip wgpu's indirect-call validation pass (orb-hal-wgpu).
    SkipIndirectValidation,
    /// Leave out `TRANSIENT_ATTACHMENT` where `GPUTextureUsage` lacks it
    /// (orb-hal-wgpu, by feature detection).
    DropTransientAttachments,
    /// The shaders avoid the miscompiled construct everywhere; a test in
    /// orb-shaders keeps it out.
    ShaderRewrite,
}

/// One known driver bug, ANGLE `FeaturesGL` style: where it occurs, how
/// Orbis meets it, and the evidence.
#[derive(Clone, Copy, Debug)]
pub struct Quirk {
    pub name: &'static str,
    /// The adapters it applies to; `None` when the workaround is applied
    /// everywhere (a shader rewrite) or by feature detection.
    pub applies: Option<fn(&Adapter<'_>) -> bool>,
    pub workaround: Workaround,
    pub evidence: &'static str,
}

/// The quirks table.
pub const QUIRKS: &[Quirk] = &[
    Quirk {
        name: "qualcomm_vulkan_indirect_validation",
        applies: Some(|a| a.api == Api::Vulkan && a.vendor == vendor::QUALCOMM),
        workaround: Workaround::SkipIndirectValidation,
        evidence: "after wgpu's indirect-call validation pass the driver draws nothing from \
                   drawIndexedIndirect (Adreno 619 measured; gfx-rs/wgpu#8801 on 650/740)",
    },
    Quirk {
        name: "adreno_function_scope_arrays",
        applies: None,
        workaround: Workaround::ShaderRewrite,
        evidence: "the Adreno 830's Vulkan driver (Chrome 156) miscompiles a compute shader \
                   that dynamically indexes function-scope arrays inlined twice: the cull wrote \
                   no draws or lost the device",
    },
    Quirk {
        name: "transient_attachment_usage",
        applies: None,
        workaround: Workaround::DropTransientAttachments,
        evidence: "Samsung Internet 30 (Chromium 143) has no GPUTextureUsage.TRANSIENT_ATTACHMENT \
                   and rejects textures with it, invalidating every frame",
    },
];

/// Whether `adapter` needs `workaround`.
pub fn needs(adapter: &Adapter<'_>, workaround: Workaround) -> bool {
    QUIRKS
        .iter()
        .any(|q| q.workaround == workaround && q.applies.is_some_and(|applies| applies(adapter)))
}

/// How long one pipeline may compile before its driver is taken to have
/// hung in its compiler: the device is then lost, and the runtime reports
/// the hang rather than make another. Set from the slowest cold compile
/// on an Adreno 619 with a wide margin.
pub const COMPILE_WATCHDOG: std::time::Duration = std::time::Duration::from_secs(60);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quirks_match_their_adapters() {
        let adreno = Adapter {
            api: Api::Vulkan,
            vendor: vendor::QUALCOMM,
            name: "Adreno (TM) 619",
            driver: "512.676",
        };
        assert!(needs(&adreno, Workaround::SkipIndirectValidation));
        let gles = Adapter {
            api: Api::Gl,
            ..adreno
        };
        assert!(!needs(&gles, Workaround::SkipIndirectValidation));
        assert!(
            !needs(&adreno, Workaround::ShaderRewrite),
            "applied everywhere"
        );
    }
}
