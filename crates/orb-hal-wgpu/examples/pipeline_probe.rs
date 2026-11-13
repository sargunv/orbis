//! Compiles one pipeline a run made, alone, as orb-hal-wgpu made it, and
//! says whether the driver took it: for finding what a driver that fails
//! to compile a pipeline ("the device was lost making it") can't take.
//! Run each try as a process of its own, since a failed compile loses the
//! device.
//!
//! ```text
//! ORB_DUMP_PIPELINES=dump orbis render <style> -o out.png --at …
//! cargo run --release -p orb-hal-wgpu --example pipeline_probe -- dump/pipelines.tsv --list
//! cargo run --release -p orb-hal-wgpu --example pipeline_probe -- dump/pipelines.tsv <label | #line> [options]
//! ```
//!
//! A label may write its spaces as `_`; it names the first line with that
//! label. `ORB_BACKEND` and wgpu's variables choose the adapter, as for the
//! map (Vulkan on Windows and macOS needs `--features vulkan`). Options:
//!
//! - `--replace OLD NEW`, `--replace-fs OLD NEW`: edit the vertex (or
//!   fragment) stage's WGSL, the first `OLD`, which must occur. The dumped
//!   WGSL has no indentation.
//! - `--nonce N`: every stage unique to `N`, as the map's shader nonce
//!   makes them (`orb_hal_wgpu::nonced`), so the system's shader caches
//!   don't hold it: a cold compile, timed after a trivial one has started
//!   the driver's compiler.
//! - `--unchecked`: modules without wgpu's runtime checks (bounds, loop
//!   bounding), to see what they cost the driver's compiler.
//! - `--specialise`: each stage's WGSL specialised to its boolean override
//!   constants, as orb-render makes it (`orb_shaders::specialise`).
//! - `--vs-entry NAME`: another vertex entry point of the same module.
//! - `--vs-const k=v,…`, `--fs-const k=v,…`: set override constants.
//! - `--const-vs`, `--const-fs`: a constant stage in its place (the vertex
//!   one gives the fragment stage its inputs, zeroed).
//! - `--spv REWRITES`: naga's SPIR-V as wgpu-hal's Vulkan back end writes
//!   it, passed straight to the driver, after the comma-separated rewrites
//!   (`none`; `split-array-extracts`, every array taken out of a composite
//!   element by element and put back together; `split-output-stores`, an
//!   array stored to an output, such as clip distances, element by element).
//!   It follows the device as wgpu-hal does (SPIR-V version, robustness,
//!   workgroup zeroing); bounds checks can be set with `--robust`,
//!   `--restrict`, `--buffer-restrict` and `--image-restrict`.
//! - `--vs-spv FILE`, `--fs-spv FILE`: SPIR-V words straight to the driver.
//! - `--emit-spv PREFIX`: write naga's SPIR-V of each stage there.
//! - `--emit-msl PREFIX`: write naga's Metal Shading Language of each stage
//!   there, about as wgpu-hal's Metal back end writes it.
//!
//! Prints `OK <label> <ms>` and exits 0, or `FAIL <label>: <why>` and exits
//! 1; 2 for bad arguments.

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    native::main();
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::borrow::Cow;
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use naga::back::spv;

    /// A stage as the dump names it, and the edits asked of its WGSL.
    struct Stage {
        file: PathBuf,
        entry: String,
        constants: Vec<(String, f64)>,
        edits: Vec<(String, String)>,
        nonce: Option<u32>,
        /// Its WGSL specialised to its boolean constants.
        specialise: bool,
    }

    impl Stage {
        fn parse(dir: &Path, fields: &[&str]) -> Self {
            Self {
                file: dir.join(fields[0]),
                entry: fields[1].to_string(),
                constants: constants(fields[2]),
                edits: Vec::new(),
                nonce: None,
                specialise: false,
            }
        }

        /// Its module as orb-shaders' build makes it: the WGSL's, cut to
        /// the one entry point.
        fn module(&self) -> Result<naga::Module, String> {
            let mut source = std::fs::read_to_string(&self.file)
                .map_err(|e| format!("{}: {e}", self.file.display()))?;
            for (old, new) in &self.edits {
                if !source.contains(old.as_str()) {
                    return Err(format!("{} has no `{old}`", self.file.display()));
                }
                source = source.replacen(old.as_str(), new, 1);
            }
            if self.specialise {
                // The variant's flags, at their defaults where unset.
                let mut constants: Vec<(&str, f64)> = (self.constants.iter())
                    .map(|(k, v)| (k.as_str(), *v))
                    .collect();
                for (name, default) in [
                    (orb_shaders::PLANE, 0.0),
                    (orb_shaders::FEATURE_ROWS, 1.0),
                    (orb_shaders::FEATURE_STATE, 1.0),
                    (orb_shaders::PATTERNS, 1.0),
                ] {
                    if !constants.iter().any(|(k, _)| *k == name) {
                        constants.push((name, default));
                    }
                }
                if let Some(specialised) = orb_shaders::specialise(&source, &constants) {
                    source = specialised;
                }
            }
            if let Some(nonce) = self.nonce {
                // A run with a nonce dumped its modules with that one.
                const NONCE: &str = " var orb_nonce: u32 = ";
                while let Some(at) = source.find(NONCE)
                    && let Some(end) = source[at..].find("u;")
                {
                    source.replace_range(at..at + end + 2, "");
                }
                source = orb_hal_wgpu::nonced(&source, nonce);
            }
            entry_module(&source, &self.entry)
        }
    }

    enum Spec {
        Render {
            label: String,
            vs: Stage,
            fs: Stage,
            targets: Vec<(wgpu::TextureFormat, Option<wgpu::BlendState>)>,
            depth: Option<(wgpu::TextureFormat, bool, bool, bool)>,
            topology: wgpu::PrimitiveTopology,
            cull: bool,
            samples: u32,
        },
        Compute {
            label: String,
            cs: Stage,
        },
    }

    impl Spec {
        fn parse(dir: &Path, line: &str) -> Self {
            let f: Vec<&str> = line.split('\t').collect();
            match f[0] {
                "render" => Self::Render {
                    label: f[1].to_string(),
                    vs: Stage::parse(dir, &f[2..5]),
                    fs: Stage::parse(dir, &f[5..8]),
                    targets: (f[8].split('|').filter(|t| !t.is_empty()))
                        .map(|t| {
                            let (format, b) = t.split_once(':').expect("format:blend");
                            (texture_format(format), blend(b))
                        })
                        .collect(),
                    depth: (f[9] != "-").then(|| {
                        let mut d = f[9].split(':');
                        let format = d.next().expect("format:write:test:once");
                        let [write, test] = [d.next(), d.next()].map(|b| b != Some("false"));
                        let once = d.next() == Some("true");
                        (texture_format(format), write, test, once)
                    }),
                    topology: match f[10] {
                        "LineList" => wgpu::PrimitiveTopology::LineList,
                        "TriangleList" => wgpu::PrimitiveTopology::TriangleList,
                        "TriangleStrip" => wgpu::PrimitiveTopology::TriangleStrip,
                        other => panic!("topology {other}"),
                    },
                    cull: f[11] == "true",
                    samples: f[12].parse().expect("a sample count"),
                },
                "compute" => Self::Compute {
                    label: f[1].to_string(),
                    cs: Stage::parse(dir, &f[2..5]),
                },
                other => panic!("pipeline kind {other}"),
            }
        }

        fn label(&self) -> &str {
            match self {
                Self::Render { label, .. } | Self::Compute { label, .. } => label,
            }
        }
    }

    /// What a stage is made from.
    enum Source {
        Naga(Box<naga::Module>, Vec<(String, f64)>),
        Spirv(Vec<u32>),
    }

    fn usage() -> ! {
        eprintln!("usage: pipeline_probe <pipelines.tsv> <label | #line | --list> [options]");
        std::process::exit(2)
    }

    pub(super) fn main() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let [tsv, which, ..] = args.as_slice() else {
            usage()
        };
        let tsv = PathBuf::from(tsv);
        let dir = tsv.parent().unwrap_or(Path::new(".")).to_path_buf();
        let text =
            std::fs::read_to_string(&tsv).unwrap_or_else(|e| panic!("{}: {e}", tsv.display()));
        let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
        if which == "--list" {
            for (i, line) in lines.iter().enumerate() {
                println!("#{i}\t{line}");
            }
            return;
        }
        let line = match which.strip_prefix('#') {
            Some(n) => n.parse::<usize>().ok().and_then(|n| lines.get(n)),
            None => {
                let label = which.replace('_', " ");
                (lines.iter()).find(|l| l.split('\t').nth(1) == Some(label.as_str()))
            }
        };
        let Some(line) = line else {
            eprintln!("pipeline_probe: no pipeline {which}");
            std::process::exit(2)
        };
        let mut spec = Spec::parse(&dir, line);

        let mut rewrites: Option<Vec<String>> = None;
        let mut robust: Option<(bool, bool)> = None;
        let (mut const_vs, mut const_fs) = (false, false);
        let (mut vs_spv, mut fs_spv, mut emit) = (None, None, None);
        let mut nonce = None;
        let mut unchecked = false;
        let mut emit_msl = None;
        let mut options = args[2..].iter();
        while let Some(option) = options.next() {
            let mut value = || options.next().cloned().unwrap_or_else(|| usage());
            match (option.as_str(), &mut spec) {
                ("--replace", Spec::Render { vs: s, .. } | Spec::Compute { cs: s, .. }) => {
                    let old = value();
                    s.edits.push((old, value()));
                }
                ("--replace-fs", Spec::Render { fs, .. }) => {
                    let old = value();
                    fs.edits.push((old, value()));
                }
                ("--nonce", any) => {
                    let n = value().parse().unwrap_or_else(|_| usage());
                    nonce = Some(n);
                    let stages: Vec<&mut Stage> = match any {
                        Spec::Render { vs, fs, .. } => vec![vs, fs],
                        Spec::Compute { cs, .. } => vec![cs],
                    };
                    for stage in stages {
                        stage.nonce = Some(n);
                    }
                }
                ("--vs-entry", Spec::Render { vs, .. }) => vs.entry = value(),
                ("--vs-const", Spec::Render { vs: s, .. } | Spec::Compute { cs: s, .. }) => {
                    merge(&mut s.constants, constants(&value()));
                }
                ("--fs-const", Spec::Render { fs, .. }) => {
                    merge(&mut fs.constants, constants(&value()));
                }
                ("--unchecked", _) => unchecked = true,
                ("--specialise", any) => {
                    let stages: Vec<&mut Stage> = match any {
                        Spec::Render { vs, fs, .. } => vec![vs, fs],
                        Spec::Compute { cs, .. } => vec![cs],
                    };
                    for stage in stages {
                        stage.specialise = true;
                    }
                }
                ("--const-vs", _) => const_vs = true,
                ("--const-fs", _) => const_fs = true,
                ("--spv", _) => {
                    rewrites = Some(
                        (value().split(',').filter(|r| *r != "none"))
                            .map(str::to_string)
                            .collect(),
                    );
                }
                ("--robust", _) => robust = Some((true, true)),
                ("--restrict", _) => robust = Some((false, false)),
                ("--buffer-restrict", _) => robust.get_or_insert((true, true)).0 = false,
                ("--image-restrict", _) => robust.get_or_insert((true, true)).1 = false,
                ("--vs-spv", _) => vs_spv = Some(value()),
                ("--fs-spv", _) => fs_spv = Some(value()),
                ("--emit-spv", _) => emit = Some(value()),
                ("--emit-msl", _) => emit_msl = Some(value()),
                _ => {
                    eprintln!("pipeline_probe: {option} isn't an option here");
                    std::process::exit(2)
                }
            }
        }

        let label = spec.label().to_string();
        let fail = |why: String| -> ! {
            println!("FAIL {label}: {why}");
            std::process::exit(1)
        };
        let read_spv = |path: &str| -> Source {
            let bytes = std::fs::read(path).unwrap_or_else(|e| fail(format!("{path}: {e}")));
            Source::Spirv(
                (bytes.as_chunks().0.iter())
                    .map(|&w| u32::from_le_bytes(w))
                    .collect(),
            )
        };
        let naga = |stage: &Stage| -> Source {
            let module = stage.module().unwrap_or_else(|e| fail(e));
            // An edit may leave a constant unread, which wgpu refuses.
            let (kept, dropped): (Vec<_>, Vec<_>) =
                (stage.constants.iter().cloned()).partition(|(name, _)| {
                    (module.overrides.iter()).any(|(_, o)| o.name.as_deref() == Some(name))
                });
            for (name, _) in dropped {
                eprintln!("pipeline_probe: {} no longer reads {name}", stage.entry);
            }
            Source::Naga(Box::new(module), kept)
        };
        let stub = |wgsl: String, entry: &str| -> Source {
            let module = entry_module(&wgsl, entry).unwrap_or_else(|e| fail(e));
            Source::Naga(Box::new(module), Vec::new())
        };
        // The stages' sources, as (source, entry point).
        let sources: Vec<(Source, String)> = match &spec {
            Spec::Render {
                vs, fs, targets, ..
            } => {
                let fs_source = match (&fs_spv, const_fs) {
                    (Some(path), _) => read_spv(path),
                    (None, true) => stub(fragment_stub(targets), "fs_stub"),
                    (None, false) => naga(fs),
                };
                let fs_entry = if const_fs && fs_spv.is_none() {
                    "fs_stub".to_string()
                } else {
                    fs.entry.clone()
                };
                let vs_source = match (&vs_spv, const_vs) {
                    (Some(path), _) => read_spv(path),
                    (None, true) => {
                        let inputs = match &fs_source {
                            Source::Naga(module, _) => Some(&**module),
                            Source::Spirv(_) => None,
                        };
                        stub(vertex_stub(inputs), "vs_stub")
                    }
                    (None, false) => naga(vs),
                };
                let vs_entry = if const_vs && vs_spv.is_none() {
                    "vs_stub".to_string()
                } else {
                    vs.entry.clone()
                };
                vec![(vs_source, vs_entry), (fs_source, fs_entry)]
            }
            Spec::Compute { cs, .. } => vec![(naga(cs), cs.entry.clone())],
        };
        let bindings = bindings_of(sources.iter().filter_map(|(s, _)| match s {
            Source::Naga(module, _) => Some(&**module),
            Source::Spirv(_) => None,
        }));

        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor
            .flags
            .remove(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL);
        let mut descriptor = descriptor.with_env();
        if let Ok(names) = std::env::var("ORB_BACKEND") {
            descriptor.backends = wgpu::Backends::from_comma_list(&names);
        }
        let instance = wgpu::Instance::new(descriptor);
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .unwrap_or_else(|e| fail(format!("no adapter: {e}")));
        let info = adapter.get_info();
        eprintln!(
            "pipeline_probe: {} ({:?}, {} {})",
            info.name, info.backend, info.driver, info.driver_info
        );
        let mut writing = writing(&adapter);
        if let Some(robust) = robust {
            writing.robust = robust;
        }
        let passthrough = rewrites.is_some() || vs_spv.is_some() || fs_spv.is_some();
        let mut features = adapter.features()
            & (wgpu::Features::PIPELINE_CACHE
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::CLIP_DISTANCES);
        if passthrough {
            features |= wgpu::Features::PASSTHROUGH_SHADERS;
        }
        let (device, _queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("pipeline probe"),
                required_features: features,
                required_limits: orb_hal_wgpu::device_limits(&adapter),
                ..Default::default()
            }))
            .unwrap_or_else(|e| fail(format!("no device: {e}")));
        let lost: Arc<Mutex<Option<String>>> = Arc::default();
        let lost_here = Arc::clone(&lost);
        device.set_device_lost_callback(move |reason, message| {
            if let Ok(mut lost) = lost_here.lock() {
                *lost = Some(format!("{reason:?}: {message}"));
            }
        });

        let binding_map = vulkan_binding_map(&bindings);
        // Each stage's module, override constants and entry point.
        type Made = (wgpu::ShaderModule, Vec<(String, f64)>, String);
        let modules: Vec<Made> = (sources.into_iter())
            .enumerate()
            .map(|(i, (source, entry))| {
                let name = ["vs", "fs"][i.min(1)];
                if let (Some(prefix), Source::Naga(module, constants)) = (&emit_msl, &source) {
                    let msl = naga_msl(module, constants).unwrap_or_else(|e| fail(e));
                    let path = format!("{prefix}-{name}.metal");
                    std::fs::write(&path, msl).unwrap_or_else(|e| fail(format!("{path}: {e}")));
                }
                if let (Some(prefix), Source::Naga(module, constants)) = (&emit, &source) {
                    let words = naga_spv(module, constants, &writing, &binding_map)
                        .unwrap_or_else(|e| fail(e));
                    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                    let path = format!("{prefix}-{name}.spv");
                    std::fs::write(&path, bytes).unwrap_or_else(|e| fail(format!("{path}: {e}")));
                }
                let words = match (source, &rewrites) {
                    (Source::Spirv(words), _) => words,
                    (Source::Naga(module, constants), Some(rewrites)) => {
                        let mut words = naga_spv(&module, &constants, &writing, &binding_map)
                            .unwrap_or_else(|e| fail(e));
                        for rewrite in rewrites {
                            words = rewrite_spv(&words, rewrite).unwrap_or_else(|e| fail(e));
                        }
                        words
                    }
                    (Source::Naga(module, constants), None) => {
                        let desc = wgpu::ShaderModuleDescriptor {
                            label: Some(name),
                            source: wgpu::ShaderSource::Naga(Cow::Owned(*module)),
                        };
                        let made = if unchecked {
                            // SAFETY: a probe's pipeline is compiled, never run.
                            unsafe {
                                device.create_shader_module_trusted(
                                    desc,
                                    wgpu::ShaderRuntimeChecks::unchecked(),
                                )
                            }
                        } else {
                            device.create_shader_module(desc)
                        };
                        return (made, constants, entry);
                    }
                };
                // SAFETY: the words are naga's valid SPIR-V, or the caller's.
                let made = unsafe {
                    device.create_shader_module_passthrough(
                        wgpu::ShaderModuleDescriptorPassthrough {
                            label: Some(name),
                            entry_points: Cow::Owned(vec![wgpu::PassthroughShaderEntryPoint {
                                name: Cow::Owned(entry.clone()),
                                workgroup_size: (1, 1, 1),
                            }]),
                            spirv: Some(Cow::Owned(words)),
                            ..Default::default()
                        },
                    )
                };
                (made, Vec::new(), entry)
            })
            .collect();

        let visibility = match spec {
            Spec::Render { .. } => wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            Spec::Compute { .. } => wgpu::ShaderStages::COMPUTE,
        };
        let layout = layout_of(&device, &bindings, visibility);
        let constants: Vec<Vec<(&str, f64)>> = (modules.iter())
            .map(|(_, c, _)| c.iter().map(|(k, v)| (k.as_str(), *v)).collect())
            .collect();
        if let Some(nonce) = nonce {
            // The driver's compiler service starts with the first compile:
            // a trivial one first, so the timed one is the pipeline's own.
            let wgsl = orb_hal_wgpu::nonced("@compute @workgroup_size(1) fn warm() {}", nonce);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("warm-up"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
            let _ = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("warm-up"),
                layout: None,
                module: &module,
                entry_point: Some("warm"),
                compilation_options: Default::default(),
                cache: None,
            });
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let start = Instant::now();
        match &spec {
            Spec::Render {
                targets,
                depth,
                topology,
                cull,
                samples,
                ..
            } => {
                let targets: Vec<_> = (targets.iter())
                    .map(|&(format, blend)| {
                        Some(wgpu::ColorTargetState {
                            format,
                            blend,
                            write_mask: wgpu::ColorWrites::ALL,
                        })
                    })
                    .collect();
                let _ = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(&label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &modules[0].0,
                        entry_point: Some(&modules[0].2),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &constants[0],
                            ..Default::default()
                        },
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState {
                        cull_mode: cull.then_some(wgpu::Face::Back),
                        topology: *topology,
                        ..Default::default()
                    },
                    depth_stencil: depth.map(|(format, write, test, once)| {
                        wgpu::DepthStencilState {
                            format,
                            depth_write_enabled: Some(write),
                            depth_compare: Some(if test {
                                wgpu::CompareFunction::GreaterEqual
                            } else {
                                wgpu::CompareFunction::Always
                            }),
                            stencil: match once {
                                true => {
                                    let face = wgpu::StencilFaceState {
                                        compare: wgpu::CompareFunction::NotEqual,
                                        fail_op: wgpu::StencilOperation::Keep,
                                        depth_fail_op: wgpu::StencilOperation::Keep,
                                        pass_op: wgpu::StencilOperation::Replace,
                                    };
                                    wgpu::StencilState {
                                        front: face,
                                        back: face,
                                        read_mask: 0xff,
                                        write_mask: 0xff,
                                    }
                                }
                                false => Default::default(),
                            },
                            bias: Default::default(),
                        }
                    }),
                    multisample: wgpu::MultisampleState {
                        count: *samples,
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &modules[1].0,
                        entry_point: Some(&modules[1].2),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &constants[1],
                            ..Default::default()
                        },
                        targets: &targets,
                    }),
                    multiview_mask: None,
                    cache: None,
                });
            }
            Spec::Compute { .. } => {
                let _ = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&label),
                    layout: Some(&layout),
                    module: &modules[0].0,
                    entry_point: Some(&modules[0].2),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants[0],
                        ..Default::default()
                    },
                    cache: None,
                });
            }
        }
        let took = start.elapsed();
        // wgpu loses the device in the call a driver fails.
        if let Some(why) = lost.lock().ok().and_then(|l| l.clone()) {
            fail(format!("the device was lost making it ({why})"));
        }
        if let Some(error) = pollster::block_on(scope.pop()) {
            fail(error.to_string());
        }
        println!("OK {label} {:.0}", took.as_secs_f64() * 1e3);
    }

    fn constants(text: &str) -> Vec<(String, f64)> {
        (text.split(',').filter(|p| !p.is_empty()))
            .map(|p| {
                let (k, v) = p.split_once('=').expect("constants as k=v");
                (k.to_string(), v.parse().expect("a number"))
            })
            .collect()
    }

    fn merge(into: &mut Vec<(String, f64)>, more: Vec<(String, f64)>) {
        for (k, v) in more {
            match into.iter_mut().find(|(name, _)| *name == k) {
                Some(slot) => slot.1 = v,
                None => into.push((k, v)),
            }
        }
    }

    fn texture_format(name: &str) -> wgpu::TextureFormat {
        use wgpu::TextureFormat as F;
        [
            F::Rgba8Unorm,
            F::Rgba8UnormSrgb,
            F::Bgra8Unorm,
            F::Bgra8UnormSrgb,
            F::Rgba16Float,
            F::R8Unorm,
            F::Rg8Unorm,
            F::R16Float,
            F::Rg16Float,
            F::R32Float,
            F::Rg32Float,
            F::R32Uint,
            F::Rg32Uint,
            F::Rgba32Float,
            F::Rgb10a2Unorm,
            F::Depth16Unorm,
            F::Depth24Plus,
            F::Depth24PlusStencil8,
            F::Depth32Float,
            F::Depth32FloatStencil8,
        ]
        .into_iter()
        .find(|f| format!("{f:?}") == name)
        .unwrap_or_else(|| panic!("format {name}"))
    }

    /// orb-hal-wgpu's blend state (`convert::color_target`) by its name.
    fn blend(name: &str) -> Option<wgpu::BlendState> {
        let component = |src, dst, operation| wgpu::BlendComponent {
            src_factor: src,
            dst_factor: dst,
            operation,
        };
        use wgpu::{BlendFactor as F, BlendOperation as O};
        match name {
            "None" => None,
            "Some(PremultipliedAlpha)" => Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            "Some(Additive)" => Some(wgpu::BlendState {
                color: component(F::One, F::One, O::Add),
                alpha: component(F::One, F::One, O::Add),
            }),
            other => panic!("blend {other}"),
        }
    }

    /// The module of `source` cut to `entry` (orb-shaders' `split`).
    fn entry_module(source: &str, entry: &str) -> Result<naga::Module, String> {
        let mut module =
            naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
        let at = (module.entry_points.iter())
            .position(|e| e.name == entry)
            .ok_or(format!("no entry point {entry}"))?;
        let entry = module.entry_points.swap_remove(at);
        module.entry_points = vec![entry];
        naga::compact::compact(&mut module, naga::compact::KeepUnused::No);
        Ok(module)
    }

    /// A fragment stage writing each target a constant.
    fn fragment_stub(targets: &[(wgpu::TextureFormat, Option<wgpu::BlendState>)]) -> String {
        if targets.is_empty() {
            return "@fragment fn fs_stub() {}".into();
        }
        let (mut members, mut values) = (String::new(), String::new());
        for (i, (format, _)) in targets.iter().enumerate() {
            let uint = matches!(
                format,
                wgpu::TextureFormat::R32Uint | wgpu::TextureFormat::Rg32Uint
            );
            let ty = if uint { "u32" } else { "f32" };
            members += &format!("@location({i}) o{i}: vec4<{ty}>, ");
            values += &format!("vec4<{ty}>({ty}(1)), ");
        }
        format!(
            "struct StubOut {{ {members} }}\n@fragment fn fs_stub() -> StubOut {{ return StubOut({values}); }}"
        )
    }

    /// A vertex stage giving `fragment` (an entry point's module) every
    /// input it reads, zeroed, and a position.
    fn vertex_stub(fragment: Option<&naga::Module>) -> String {
        let mut members = vec!["@builtin(position) position: vec4<f32>,".to_string()];
        if let Some(fragment) = fragment {
            let mut add = |binding: &Option<naga::Binding>, ty: naga::Handle<naga::Type>| {
                let Some(naga::Binding::Location {
                    location,
                    interpolation,
                    sampling,
                    ..
                }) = binding
                else {
                    return;
                };
                let scalar = |s: naga::Scalar| match s.kind {
                    naga::ScalarKind::Uint => "u32",
                    naga::ScalarKind::Sint => "i32",
                    _ => "f32",
                };
                let ty = match fragment.types[ty].inner {
                    naga::TypeInner::Scalar(s) => scalar(s).to_string(),
                    naga::TypeInner::Vector { size, scalar: s } => {
                        format!("vec{}<{}>", size as u8, scalar(s))
                    }
                    ref other => panic!("an input of {other:?}"),
                };
                let lower = |d: &dyn std::fmt::Debug| format!("{d:?}").to_lowercase();
                let interpolate = match (interpolation, sampling) {
                    (Some(naga::Interpolation::Flat), Some(naga::Sampling::Either)) => {
                        "@interpolate(flat, either) ".to_string()
                    }
                    (Some(naga::Interpolation::Flat), _) => "@interpolate(flat) ".to_string(),
                    (Some(i), Some(s)) => format!("@interpolate({}, {}) ", lower(i), lower(s)),
                    (Some(i), None) => format!("@interpolate({}) ", lower(i)),
                    (None, _) => String::new(),
                };
                members.push(format!(
                    "@location({location}) {interpolate}m{location}: {ty},"
                ));
            };
            for argument in &fragment.entry_points[0].function.arguments {
                match &fragment.types[argument.ty].inner {
                    naga::TypeInner::Struct { members, .. } => {
                        for member in members {
                            add(&member.binding, member.ty);
                        }
                    }
                    _ => add(&argument.binding, argument.ty),
                }
            }
        }
        format!(
            "struct StubOut {{ {} }}\n@vertex fn vs_stub() -> StubOut {{ var out: StubOut; out.position = vec4<f32>(0.0, 0.0, 0.5, 1.0); return out; }}",
            members.join(" ")
        )
    }

    type Bindings = BTreeMap<u32, BTreeMap<u32, wgpu::BindingType>>;

    /// Every binding `modules` declare, by group and binding.
    fn bindings_of<'a>(modules: impl Iterator<Item = &'a naga::Module>) -> Bindings {
        let mut groups = Bindings::new();
        for module in modules {
            for (_, var) in module.global_variables.iter() {
                let Some(at) = &var.binding else { continue };
                let ty = match (var.space, &module.types[var.ty].inner) {
                    (naga::AddressSpace::Uniform, _) => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    (naga::AddressSpace::Storage { access }, _) => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: !access.contains(naga::StorageAccess::STORE),
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    (_, &naga::TypeInner::Sampler { comparison }) => {
                        wgpu::BindingType::Sampler(if comparison {
                            wgpu::SamplerBindingType::Comparison
                        } else {
                            wgpu::SamplerBindingType::Filtering
                        })
                    }
                    (
                        _,
                        &naga::TypeInner::Image {
                            dim,
                            arrayed,
                            class,
                        },
                    ) => image_binding(dim, arrayed, class),
                    (_, other) => panic!("a binding of {other:?}"),
                };
                groups.entry(at.group).or_default().insert(at.binding, ty);
            }
        }
        groups
    }

    fn image_binding(
        dim: naga::ImageDimension,
        arrayed: bool,
        class: naga::ImageClass,
    ) -> wgpu::BindingType {
        use naga::ImageDimension as D;
        use wgpu::TextureViewDimension as V;
        let view_dimension = match (dim, arrayed) {
            (D::D1, _) => V::D1,
            (D::D2, false) => V::D2,
            (D::D2, true) => V::D2Array,
            (D::D3, _) => V::D3,
            (D::Cube, false) => V::Cube,
            (D::Cube, true) => V::CubeArray,
        };
        match class {
            naga::ImageClass::Sampled { kind, multi } => wgpu::BindingType::Texture {
                sample_type: match kind {
                    naga::ScalarKind::Uint => wgpu::TextureSampleType::Uint,
                    naga::ScalarKind::Sint => wgpu::TextureSampleType::Sint,
                    _ => wgpu::TextureSampleType::Float { filterable: !multi },
                },
                view_dimension,
                multisampled: multi,
            },
            naga::ImageClass::Depth { multi } => wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension,
                multisampled: multi,
            },
            naga::ImageClass::Storage { format, access } => wgpu::BindingType::StorageTexture {
                access: match (
                    access.contains(naga::StorageAccess::LOAD),
                    access.contains(naga::StorageAccess::STORE),
                ) {
                    (true, true) => wgpu::StorageTextureAccess::ReadWrite,
                    (false, true) => wgpu::StorageTextureAccess::WriteOnly,
                    _ => wgpu::StorageTextureAccess::ReadOnly,
                },
                format: match format {
                    naga::StorageFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
                    naga::StorageFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
                    naga::StorageFormat::R32Float => wgpu::TextureFormat::R32Float,
                    naga::StorageFormat::R32Uint => wgpu::TextureFormat::R32Uint,
                    naga::StorageFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
                    naga::StorageFormat::Rgba32Uint => wgpu::TextureFormat::Rgba32Uint,
                    other => panic!("a storage texture of {other:?}"),
                },
                view_dimension,
            },
            other => panic!("an image of {other:?}"),
        }
    }

    /// One bind group layout per group up to the last, each binding
    /// visible to `visibility`: passthrough stages can't make their own.
    fn layout_of(
        device: &wgpu::Device,
        groups: &Bindings,
        visibility: wgpu::ShaderStages,
    ) -> wgpu::PipelineLayout {
        let count = groups.keys().next_back().map_or(0, |g| g + 1);
        let layouts: Vec<wgpu::BindGroupLayout> = (0..count)
            .map(|group| {
                let entries: Vec<_> = (groups.get(&group).into_iter().flatten())
                    .map(|(&binding, &ty)| wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility,
                        ty,
                        count: None,
                    })
                    .collect();
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &entries,
                })
            })
            .collect();
        let layouts: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(Some).collect();
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline probe"),
            bind_group_layouts: &layouts,
            immediate_size: 0,
        })
    }

    /// wgpu-hal's Vulkan back end numbers each set's bindings densely, in
    /// order, and writes SPIR-V with this map.
    fn vulkan_binding_map(groups: &Bindings) -> spv::BindingMap {
        let mut map = spv::BindingMap::default();
        for (&group, bindings) in groups {
            for (index, &binding) in bindings.keys().enumerate() {
                map.insert(
                    naga::ResourceBinding { group, binding },
                    spv::BindingInfo {
                        descriptor_set: group,
                        binding: index as u32,
                        binding_array_size: None,
                    },
                );
            }
        }
        map
    }

    /// How wgpu-hal's Vulkan back end writes SPIR-V for a device.
    struct Writing {
        /// Whether buffer and image bounds are left to the device
        /// (`robustBufferAccess2`, robust image access).
        robust: (bool, bool),
        /// Whether workgroup memory is zeroed by the driver.
        zero_init: bool,
        lang_version: (u8, u8),
    }

    /// How wgpu-hal writes SPIR-V for `adapter`, read as it reads it
    /// (`naga_options`); SPIR-V 1.6 with everything left to the device off
    /// Vulkan.
    #[cfg(any(target_os = "linux", target_os = "android", feature = "vulkan"))]
    fn writing(adapter: &wgpu::Adapter) -> Writing {
        use ash::vk;
        // SAFETY: only reads the adapter's Vulkan objects.
        let Some(hal) = (unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }) else {
            return writing_elsewhere();
        };
        let capabilities = hal.physical_device_capabilities();
        let api = capabilities.properties().api_version;
        let robustness2 = capabilities.supports_extension(ash::ext::robustness2::NAME);
        let image_robustness = api >= vk::API_VERSION_1_3
            || capabilities.supports_extension(ash::ext::image_robustness::NAME);
        let zero_init = api >= vk::API_VERSION_1_3
            || capabilities.supports_extension(ash::khr::zero_initialize_workgroup_memory::NAME);
        let mut r2 = vk::PhysicalDeviceRobustness2FeaturesEXT::default();
        let mut image = vk::PhysicalDeviceImageRobustnessFeatures::default();
        let mut zero = vk::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures::default();
        let mut features = vk::PhysicalDeviceFeatures2::default();
        if robustness2 {
            features = features.push_next(&mut r2);
        }
        if image_robustness {
            features = features.push_next(&mut image);
        }
        if zero_init {
            features = features.push_next(&mut zero);
        }
        // SAFETY: the structs chained are of extensions the device has.
        unsafe {
            (hal.shared_instance().raw_instance())
                .get_physical_device_features2(hal.raw_physical_device(), &mut features);
        }
        let buffer = robustness2 && r2.robust_buffer_access2 == vk::TRUE;
        let image = if robustness2 {
            r2.robust_image_access2 == vk::TRUE
        } else {
            image_robustness && image.robust_image_access == vk::TRUE
        };
        let lang_version = match api {
            vk::API_VERSION_1_3.. => (1, 6),
            vk::API_VERSION_1_2.. => (1, 5),
            vk::API_VERSION_1_1.. => (1, 3),
            _ => (1, 0),
        };
        Writing {
            robust: (buffer, image),
            // Polyfilled on MoltenVK, whose SPIRV-Cross can't lower it.
            zero_init: zero_init
                && zero.shader_zero_initialize_workgroup_memory == vk::TRUE
                && !capabilities.is_driver(vk::DriverId::MOLTENVK),
            lang_version,
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "android", feature = "vulkan")))]
    fn writing(_: &wgpu::Adapter) -> Writing {
        writing_elsewhere()
    }

    fn writing_elsewhere() -> Writing {
        Writing {
            robust: (true, true),
            zero_init: true,
            lang_version: (1, 6),
        }
    }

    /// naga's SPIR-V of a stage as wgpu-hal's Vulkan back end writes it
    /// (`naga_options`, `compile_stage`), without varying names as for
    /// Qualcomm's drivers.
    /// `module`'s one entry point in Metal Shading Language, its override
    /// constants `constants`, with wgpu-hal's bounds checks.
    fn naga_msl(module: &naga::Module, constants: &[(String, f64)]) -> Result<String, String> {
        use naga::back::msl;
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(module)
        .map_err(|e| format!("{e:?}"))?;
        let entry = &module.entry_points[0];
        let map = constants.iter().cloned().collect();
        let (module, info) = naga::back::pipeline_constants::process_overrides(
            module,
            &info,
            Some((entry.stage, entry.name.as_str())),
            &map,
        )
        .map_err(|e| format!("{e:?}"))?;
        let check = naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite;
        let options = msl::Options {
            lang_version: (3, 0),
            bounds_check_policies: naga::proc::BoundsCheckPolicies {
                index: check,
                buffer: check,
                image_load: check,
                binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
            },
            ..msl::Options::default()
        };
        let pipeline = msl::PipelineOptions {
            entry_point: Some((entry.stage, entry.name.clone())),
            vertex_pulling_transform: true,
            ..msl::PipelineOptions::default()
        };
        msl::write_string(&module, &info, &options, &pipeline)
            .map(|(source, _)| source)
            .map_err(|e| format!("{e:?}"))
    }

    fn naga_spv(
        module: &naga::Module,
        constants: &[(String, f64)],
        writing: &Writing,
        binding_map: &spv::BindingMap,
    ) -> Result<Vec<u32>, String> {
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(module)
        .map_err(|e| format!("{e:?}"))?;
        let entry = &module.entry_points[0];
        let map = constants.iter().cloned().collect();
        let (module, info) = naga::back::pipeline_constants::process_overrides(
            module,
            &info,
            Some((entry.stage, entry.name.as_str())),
            &map,
        )
        .map_err(|e| format!("{e:?}"))?;
        let policy = |robust| match robust {
            true => naga::proc::BoundsCheckPolicy::Unchecked,
            false => naga::proc::BoundsCheckPolicy::Restrict,
        };
        let options = spv::Options {
            lang_version: writing.lang_version,
            flags: spv::WriterFlags::FORCE_POINT_SIZE,
            bounds_check_policies: naga::proc::BoundsCheckPolicies {
                index: naga::proc::BoundsCheckPolicy::Restrict,
                buffer: policy(writing.robust.0),
                image_load: policy(writing.robust.1),
                binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
            },
            zero_initialize_workgroup_memory: if writing.zero_init {
                spv::ZeroInitializeWorkgroupMemoryMode::Native
            } else {
                spv::ZeroInitializeWorkgroupMemoryMode::Polyfill
            },
            force_loop_bounding: true,
            binding_map: binding_map.clone(),
            ..Default::default()
        };
        let entry = &module.entry_points[0];
        let pipeline = spv::PipelineOptions {
            shader_stage: entry.stage,
            entry_point: entry.name.clone(),
        };
        spv::write_vec(&module, &info, &options, Some(&pipeline)).map_err(|e| format!("{e:?}"))
    }

    /// `words` rewritten as `rewrite` names (see the options above).
    fn rewrite_spv(words: &[u32], rewrite: &str) -> Result<Vec<u32>, String> {
        const TYPE_INT: u32 = 21;
        const TYPE_ARRAY: u32 = 28;
        const TYPE_POINTER: u32 = 32;
        const CONSTANT: u32 = 43;
        const VARIABLE: u32 = 59;
        const STORE: u32 = 62;
        const ACCESS_CHAIN: u32 = 65;
        const COMPOSITE_CONSTRUCT: u32 = 80;
        const COMPOSITE_EXTRACT: u32 = 81;
        const OUTPUT: u32 = 3;
        let instructions = || {
            let mut at = 5;
            std::iter::from_fn(move || {
                let word = *words.get(at)?;
                let n = (word >> 16) as usize;
                let instruction = &words[at..at + n];
                at += n;
                Some((word & 0xffff, instruction))
            })
        };
        // id → (element type, length); id → (type, value); (id, storage, pointee); variable → pointer type.
        let mut arrays = BTreeMap::new();
        let mut constants = BTreeMap::new();
        let mut pointers = Vec::new();
        let mut variables = BTreeMap::new();
        let mut uint = None;
        for (op, w) in instructions() {
            match op {
                TYPE_INT if w[2] == 32 && w[3] == 0 => uint = Some(w[1]),
                TYPE_ARRAY => drop(arrays.insert(w[1], (w[2], w[3]))),
                CONSTANT => drop(constants.insert(w[2], (w[1], w[3]))),
                TYPE_POINTER => pointers.push((w[1], w[2], w[3])),
                VARIABLE => drop(variables.insert(w[2], w[1])),
                _ => {}
            }
        }
        let array = |ty: u32| {
            let &(element, length) = arrays.get(&ty)?;
            Some((element, constants.get(&length)?.1))
        };
        let uint_constant = |value: u32| {
            (constants.iter())
                .find(|&(_, &(ty, v))| Some(ty) == uint && v == value)
                .map(|(&id, _)| id)
        };
        let mut bound = words[3];
        let mut out = words[..5].to_vec();
        let mut rewritten = 0;
        for (op, w) in instructions() {
            match rewrite {
                "split-array-extracts" if op == COMPOSITE_EXTRACT => {
                    if let Some((element, length)) = array(w[1]) {
                        let mut parts = Vec::new();
                        for k in 0..length {
                            out.push(((w.len() as u32 + 1) << 16) | COMPOSITE_EXTRACT);
                            out.extend([element, bound]);
                            out.extend(&w[3..]);
                            out.push(k);
                            parts.push(bound);
                            bound += 1;
                        }
                        out.extend([((3 + length) << 16) | COMPOSITE_CONSTRUCT, w[1], w[2]]);
                        out.extend(parts);
                        rewritten += 1;
                        continue;
                    }
                }
                "split-output-stores" if op == STORE => {
                    let output = (variables.get(&w[1]))
                        .and_then(|ty| pointers.iter().find(|p| p.0 == *ty))
                        .filter(|p| p.1 == OUTPUT)
                        .and_then(|p| array(p.2));
                    if let Some((element, length)) = output
                        && let Some(&(element_pointer, ..)) =
                            pointers.iter().find(|p| p.1 == OUTPUT && p.2 == element)
                        && let Some(indices) =
                            (0..length).map(uint_constant).collect::<Option<Vec<_>>>()
                    {
                        for (k, index) in indices.into_iter().enumerate() {
                            let (value, pointer) = (bound, bound + 1);
                            bound += 2;
                            out.extend([
                                (5 << 16) | COMPOSITE_EXTRACT,
                                element,
                                value,
                                w[2],
                                k as u32,
                            ]);
                            out.extend([
                                (5 << 16) | ACCESS_CHAIN,
                                element_pointer,
                                pointer,
                                w[1],
                                index,
                            ]);
                            out.extend([(3 << 16) | STORE, pointer, value]);
                        }
                        rewritten += 1;
                        continue;
                    }
                }
                "split-array-extracts" | "split-output-stores" => {}
                other => return Err(format!("no rewrite {other}")),
            }
            out.extend_from_slice(w);
        }
        out[3] = bound;
        eprintln!("pipeline_probe: {rewrite}: {rewritten} rewritten");
        Ok(out)
    }
}
