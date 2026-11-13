//! The runtime's GPU: a wgpu device on the default adapter.

use orb_hal_wgpu::WgpuGpu;

use crate::Error;

/// A fault tests inject: `hang`, compiles never finish, as in a driver
/// hung in its compiler; `slow:<label>[,<label>…]:<ms>`, compiles of
/// pipelines of those labels take that much longer.
const FAULT: &str = "ORB_FAULT_INJECTION";

/// A device on the default adapter, and which adapter it is: a phrase for
/// diagnostics.
pub(crate) fn make() -> Result<(WgpuGpu, String), Error> {
    let gpu = pollster::block_on(WgpuGpu::offscreen()).map_err(Error::Native)?;
    let info = gpu
        .adapter_info()
        .expect("offscreen devices keep their adapter");
    let fault = std::env::var(FAULT).unwrap_or_default();
    if let Some((label, ms)) = (fault.strip_prefix("slow:"))
        .and_then(|f| f.rsplit_once(':'))
        .and_then(|(label, ms)| Some((label, ms.parse().ok()?)))
    {
        let labels: Vec<&str> = label.split(',').collect();
        gpu.inject_slow_compile(&labels, std::time::Duration::from_millis(ms));
    }
    if fault == "hang" {
        gpu.inject_compile_hang();
        if let Some(ms) = std::env::var("ORB_COMPILE_WATCHDOG_MS")
            .ok()
            .and_then(|ms| ms.parse().ok())
        {
            gpu.set_compile_watchdog(std::time::Duration::from_millis(ms));
        }
    }
    let compiler = gpu
        .shader_compiler()
        .map(|c| format!(", shaders by {c}"))
        .unwrap_or_default();
    Ok((gpu, format!("{} ({:?}{compiler})", info.name, info.backend)))
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
