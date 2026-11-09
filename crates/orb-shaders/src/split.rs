// A module per entry point, shared by the build script and the tests.

/// `module` as one module per entry point, each without what that entry
/// point doesn't use: what wgpu's pipeline creation makes of a stage's
/// module (`naga::back::pipeline_constants::process_overrides`), which it
/// skips for a module of one entry point.
pub(crate) fn entry_modules(module: &naga::Module) -> Vec<(String, naga::Module)> {
    (0..module.entry_points.len())
        .map(|e| {
            let mut one = module.clone();
            let entry = one.entry_points.swap_remove(e);
            one.entry_points = vec![entry];
            naga::compact::compact(&mut one, naga::compact::KeepUnused::No);
            (one.entry_points[0].name.clone(), one)
        })
        .collect()
}
