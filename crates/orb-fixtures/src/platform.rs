//! The platforms a suite runs on, by `--platform`'s names, and what each
//! one is: a backend, the platform it runs beside to say what it alone
//! fails, and what it needs built. xtask's `rt` reads this file too (by
//! `#[path]`), for what to build before it starts the runner.

use std::fmt;
use std::str::FromStr;

/// One platform a suite runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Platform {
    /// The host's wgpu backend (Metal, Vulkan or D3D12).
    Native,
    /// Native again in a process on WebGL 2's CPU frame driver, its tables
    /// in storage buffers.
    NativeCpu,
    /// The same with its tables in data textures.
    NativeCpuTex,
    /// The CPU driver with everything else WebGL 2 lacks too (compute, so
    /// drawn mips, and its gaps).
    NativeWebgl2,
    /// wgpu's GLES back end, through ANGLE on macOS.
    NativeGl,
    /// The native runner in a Linux container on Mesa's lavapipe.
    Lavapipe,
    /// This machine's headless Chromium on WebGPU.
    Chrome,
    /// The same Chromium without WebGPU, on WebGL 2.
    ChromeWebgl2,
    /// The assigned phone's Chrome on WebGPU.
    AndroidChrome,
    /// The assigned phone's Chrome on WebGL 2.
    AndroidChromeWebgl2,
    /// The installed Firefox, headed, on WebGPU.
    Firefox,
    /// Playwright's WebKit on WebGPU.
    Webkit,
}

/// What `--platform` picks: one platform with what it runs beside, or
/// native, Chrome and lavapipe together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Selection {
    One(Platform),
    All,
}

/// The browser a browser platform draws in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Browser {
    Chromium,
    /// The assigned phone's Chrome.
    Android,
    Firefox,
    Webkit,
}

/// Where a beside-native platform's process draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Device {
    /// On WebGL 2's CPU frame driver, by `ORB_DRIVER`'s name for it.
    Driver(&'static str),
    /// On wgpu's GLES back end.
    Gl,
}

impl Platform {
    /// Every platform, in `--platform`'s usage order.
    pub(crate) const ALL: [Platform; 12] = [
        Platform::Native,
        Platform::Chrome,
        Platform::ChromeWebgl2,
        Platform::AndroidChrome,
        Platform::AndroidChromeWebgl2,
        Platform::Firefox,
        Platform::Webkit,
        Platform::Lavapipe,
        Platform::NativeCpu,
        Platform::NativeCpuTex,
        Platform::NativeWebgl2,
        Platform::NativeGl,
    ];

    /// Its name for `--platform`, and its report directory.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Platform::Native => "native",
            Platform::NativeCpu => "native-cpu",
            Platform::NativeCpuTex => "native-cpu-tex",
            Platform::NativeWebgl2 => "native-webgl2",
            Platform::NativeGl => "native-gl",
            Platform::Lavapipe => "lavapipe",
            Platform::Chrome => "chrome",
            Platform::ChromeWebgl2 => "chrome-webgl2",
            Platform::AndroidChrome => "android-chrome",
            Platform::AndroidChromeWebgl2 => "android-chrome-webgl2",
            Platform::Firefox => "firefox",
            Platform::Webkit => "webkit",
        }
    }

    /// The platform it runs beside in one invocation, for what it alone
    /// draws otherwise: native for Chrome (the identity check) and for the
    /// beside-native platforms (byte for byte); Chrome on WebGPU for
    /// Chrome on WebGL 2, Firefox and WebKit; this machine's Chrome on the
    /// same backend for a phone's.
    pub(crate) fn reference(self) -> Option<Platform> {
        match self {
            Platform::Native | Platform::Lavapipe => None,
            Platform::Chrome
            | Platform::NativeCpu
            | Platform::NativeCpuTex
            | Platform::NativeWebgl2
            | Platform::NativeGl => Some(Platform::Native),
            Platform::ChromeWebgl2
            | Platform::AndroidChrome
            | Platform::Firefox
            | Platform::Webkit => Some(Platform::Chrome),
            Platform::AndroidChromeWebgl2 => Some(Platform::ChromeWebgl2),
        }
    }

    /// The browser it draws in, and whether on WebGL 2; `None` natively.
    pub(crate) fn browser(self) -> Option<(Browser, bool)> {
        match self {
            Platform::Chrome => Some((Browser::Chromium, false)),
            Platform::ChromeWebgl2 => Some((Browser::Chromium, true)),
            Platform::AndroidChrome => Some((Browser::Android, false)),
            Platform::AndroidChromeWebgl2 => Some((Browser::Android, true)),
            Platform::Firefox => Some((Browser::Firefox, false)),
            Platform::Webkit => Some((Browser::Webkit, false)),
            _ => None,
        }
    }

    /// Where a platform that runs native's suite again in a process of
    /// its own draws; `None` for the rest.
    pub(crate) fn beside_native(self) -> Option<Device> {
        match self {
            Platform::NativeCpu => Some(Device::Driver("cpu")),
            Platform::NativeCpuTex => Some(Device::Driver("cpu-tex")),
            Platform::NativeWebgl2 => Some(Device::Driver("webgl2")),
            Platform::NativeGl => Some(Device::Gl),
            _ => None,
        }
    }
}

impl Selection {
    /// Every platform it runs, its references first.
    pub(crate) fn platforms(self) -> Vec<Platform> {
        match self {
            Selection::All => vec![Platform::Native, Platform::Chrome, Platform::Lavapipe],
            Selection::One(platform) => {
                platform.reference().into_iter().chain([platform]).collect()
            }
        }
    }

    /// `--platform`'s usage: every name, `|`-separated.
    pub(crate) fn usage() -> String {
        (Platform::ALL.iter().map(|p| p.name()))
            .chain(["all"])
            .collect::<Vec<_>>()
            .join("|")
    }
}

impl fmt::Display for Platform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.name())
    }
}

impl FromStr for Platform {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, String> {
        (Platform::ALL.into_iter())
            .find(|p| p.name() == name)
            .ok_or_else(|| format!("unknown platform {name}"))
    }
}

impl FromStr for Selection {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, String> {
        match name {
            "all" => Ok(Selection::All),
            _ => name.parse().map(Selection::One),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_platform_parses_back_from_its_name() {
        for platform in Platform::ALL {
            assert_eq!(platform.name().parse(), Ok(platform));
        }
        assert_eq!("all".parse(), Ok(Selection::All));
        assert!("chromium".parse::<Selection>().is_err());
    }

    #[test]
    fn a_platform_pads_to_a_table_column() {
        assert_eq!(format!("{:<8}|", Platform::Native), "native  |");
    }
}
