//! The runtime's GPU: a wgpu device on the default adapter.

use orb_hal_wgpu::WgpuGpu;

use crate::Error;

/// A fault tests inject: `hang`, compiles never finish, as in a driver
/// hung in its compiler.
const FAULT: &str = "ORB_FAULT_INJECTION";

/// A device on the default adapter, and which adapter it is: a phrase for
/// diagnostics.
pub(crate) fn make() -> Result<(WgpuGpu, String), Error> {
    let gpu = pollster::block_on(WgpuGpu::offscreen()).map_err(Error::Native)?;
    let info = gpu
        .adapter_info()
        .expect("offscreen devices keep their adapter");
    if std::env::var_os(FAULT).is_some_and(|f| f == "hang") {
        gpu.inject_compile_hang();
        if let Some(ms) = std::env::var("ORB_COMPILE_WATCHDOG_MS")
            .ok()
            .and_then(|ms| ms.parse().ok())
        {
            gpu.set_compile_watchdog(std::time::Duration::from_millis(ms));
        }
    }
    Ok((gpu, format!("{} ({:?})", info.name, info.backend)))
}

/// Compiles `gpu`'s pipelines on `executor`, calling `wake` whenever one
/// compiles.
pub(crate) fn set_compiler(
    gpu: &mut WgpuGpu,
    executor: std::sync::Arc<dyn orb_host::jobs::Executor>,
    wake: std::sync::Arc<dyn Fn() + Send + Sync>,
) {
    gpu.set_compile_executor(Some(executor));
    gpu.set_compile_wake(wake);
}
