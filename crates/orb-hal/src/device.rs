//! Device policy (plan §8.8): which GPUs get the Modern tier, the quirks
//! table, and the crash sentinel's decision. Data and pure logic: hosts
//! bring the adapter's facts and keep the sentinel's marker.

/// PCI vendor ids, as Vulkan, Metal and WebGPU report them.
pub mod vendor {
    pub const QUALCOMM: u32 = 0x5143;
    pub const ARM: u32 = 0x13b5;
    pub const IMAGINATION: u32 = 0x1010;
    pub const INTEL: u32 = 0x8086;
    pub const SAMSUNG: u32 = 0x144d;
}

/// What a host knows about a GPU when choosing its tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Adapter<'a> {
    pub platform: Platform,
    pub api: Api,
    pub vendor: u32,
    pub name: &'a str,
    pub driver: &'a str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Platform {
    Android { api_level: u32 },
    Apple,
    Windows,
    Linux,
    Web,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    Vulkan,
    Metal,
    Dx12,
    WebGpu,
    Gl,
}

/// The Android allowlist (plan §8.8 1): Vulkan is Modern only on Android
/// 12+ with GPUs and drivers known to work, seeded from Chrome's WebGPU
/// rollout on Android (Qualcomm, ARM and Intel from 12, Imagination from
/// 16, Samsung Xclipse pending). Everything else draws on GLES (Compat).
/// It grows from farm and field reports only. `Err` says why not.
pub fn android_vulkan_allowed(api_level: u32, vendor_id: u32) -> Result<(), &'static str> {
    const ANDROID_12: u32 = 31;
    const ANDROID_16: u32 = 36;
    if api_level < ANDROID_12 {
        return Err("Vulkan is Modern from Android 12");
    }
    match vendor_id {
        vendor::QUALCOMM | vendor::ARM | vendor::INTEL => Ok(()),
        vendor::IMAGINATION if api_level >= ANDROID_16 => Ok(()),
        vendor::IMAGINATION => Err("Imagination GPUs are Modern from Android 16"),
        vendor::SAMSUNG => Err("Samsung Xclipse GPUs are pending"),
        _ => Err("this GPU is not on the allowlist"),
    }
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

/// The quirks table (plan §8.8 2).
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
                   no draws or lost the device (#23)",
    },
    Quirk {
        name: "transient_attachment_usage",
        applies: None,
        workaround: Workaround::DropTransientAttachments,
        evidence: "Samsung Internet 30 (Chromium 143) has no GPUTextureUsage.TRANSIENT_ATTACHMENT \
                   and rejects textures with it, invalidating every frame (#23)",
    },
];

/// Whether `adapter` needs `workaround`.
pub fn needs(adapter: &Adapter<'_>, workaround: Workaround) -> bool {
    QUIRKS
        .iter()
        .any(|q| q.workaround == workaround && q.applies.is_some_and(|applies| applies(adapter)))
}

/// The crash sentinel (plan §8.8 3): a marker naming the device is written
/// before the first pipelines are made and cleared once they are. A marker
/// found at the next start for the same device means its driver crashed
/// compiling them, so it draws on Compat; a driver update (a new key)
/// tries Modern again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Sentinel {
    /// Nothing recorded: the device starts on its usual tier.
    Clear,
    /// The device crashed making pipelines last time: start on Compat.
    Tripped,
}

impl Sentinel {
    /// What a marker found at start (`None` without one) says about the
    /// device `key` names.
    pub fn read(marker: Option<&str>, key: &str) -> Self {
        match marker {
            Some(found) if found == key => Self::Tripped,
            _ => Self::Clear,
        }
    }
}

/// How long one pipeline may compile before its driver is taken to have
/// hung in its compiler (plan §8.8 3): the sentinel then trips as for a
/// crash, and the device is lost. Set from M4's slowest cold compile on an
/// Adreno 619 (decision 0038) with a wide margin.
pub const COMPILE_WATCHDOG: std::time::Duration = std::time::Duration::from_secs(60);

/// A sentinel key for a device: its identity and driver.
pub fn sentinel_key(adapter: &Adapter<'_>) -> String {
    format!(
        "{:?} {:?} {:#06x} {} {}",
        adapter.platform, adapter.api, adapter.vendor, adapter.name, adapter.driver
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowlist_follows_chromes_rollout() {
        assert!(android_vulkan_allowed(31, vendor::QUALCOMM).is_ok());
        assert!(android_vulkan_allowed(34, vendor::ARM).is_ok());
        assert!(
            android_vulkan_allowed(30, vendor::QUALCOMM).is_err(),
            "Android 11"
        );
        assert!(android_vulkan_allowed(35, vendor::IMAGINATION).is_err());
        assert!(android_vulkan_allowed(36, vendor::IMAGINATION).is_ok());
        assert!(android_vulkan_allowed(36, vendor::SAMSUNG).is_err());
        assert!(android_vulkan_allowed(36, 0x1234).is_err());
    }

    #[test]
    fn quirks_match_their_adapters() {
        let adreno = Adapter {
            platform: Platform::Android { api_level: 35 },
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

    #[test]
    fn sentinel_trips_for_the_same_device_only() {
        assert_eq!(Sentinel::read(None, "a"), Sentinel::Clear);
        assert_eq!(Sentinel::read(Some("a"), "a"), Sentinel::Tripped);
        assert_eq!(
            Sentinel::read(Some("a"), "b"),
            Sentinel::Clear,
            "a new driver retries Modern"
        );
    }
}
