//! The pinned expression fixtures through the GPU interpreter, against the
//! VM (decision 0017): natively and in Chrome, with the same Rust body.
//!
//! Every fixture that compiles on the CPU and lowers to a [`Program`] runs
//! each of its inputs once in a compute dispatch of `expr.wgsl`; the result
//! must equal the VM's: the same tag, the same booleans, null and errors
//! exactly, numbers and colours within f32 output precision.

use std::collections::BTreeMap;

use crate::conformance::{FIXTURES, compile, evaluate, fixture_spec, json_feature, state, zoom};
use crate::eval::Feature;
use crate::gpu::{Fallback, Layout, Program, Source};
use crate::json::{self, Json};
use crate::value::Value;

/// One fixture input to run.
struct Case {
    fixture: String,
    input: usize,
    want: Result<Value<'static>, ()>,
    /// Program, inputs, state (or `u32::MAX`), zoom f32, zoom f64 bits.
    row: [u32; 8],
}

#[derive(Default)]
struct Counts {
    fixtures: usize,
    compiled: usize,
    evaluable: usize,
    fallbacks: BTreeMap<Fallback, usize>,
}

/// Branches f32 would take differently from the VM: inputs and literals
/// that round to the same f32, beyond 2^24 and 2^32, and a zoom just
/// below a stop. Each is `expression`, `inputs` as in the fixtures.
const PROBES: &str = r#"{
  "probe/equal-near": {"expression": ["==", ["get", "x"], 0.1],
    "inputs": [[{}, {"properties": {"x": 0.1}}], [{}, {"properties": {"x": 0.10000000000000002}}],
               [{}, {"properties": {"x": 0.100000001}}]]},
  "probe/less-2^24": {"expression": ["<", ["get", "x"], 16777217],
    "inputs": [[{}, {"properties": {"x": 16777216}}], [{}, {"properties": {"x": 16777217}}],
               [{}, {"properties": {"x": 16777216.5}}]]},
  "probe/match-2^32": {"expression": ["match", ["get", "n"], 4294967297, 1, 4294967296, 2, 0],
    "propertySpec": {"type": "number"},
    "inputs": [[{}, {"properties": {"n": 4294967296}}], [{}, {"properties": {"n": 4294967297}}],
               [{}, {"properties": {"n": 4294967298}}]]},
  "probe/step-tenth": {"expression": ["step", ["get", "x"], 0, 0.1, 1],
    "propertySpec": {"type": "number"},
    "inputs": [[{}, {"properties": {"x": 0.09999999999}}], [{}, {"properties": {"x": 0.1}}],
               [{}, {"properties": {"x": -0.0}}]]},
  "probe/zoom-step": {"expression": ["step", ["zoom"], 0, 10.5, 1],
    "propertySpec": {"type": "number", "expression": {"parameters": ["zoom"]}},
    "inputs": [[{"zoom": 10.499999999}, {}], [{"zoom": 10.5}, {}]]},
  "probe/in-near": {"expression": ["in", ["get", "x"], ["literal", [0.3, "a", true]]],
    "inputs": [[{}, {"properties": {"x": 0.30000000000000004}}], [{}, {"properties": {"x": 0.3}}],
               [{}, {"properties": {"x": "a"}}], [{}, {"properties": {"x": "b"}}]]},
  "probe/state-boolean": {"expression": ["case", ["boolean", ["feature-state", "hover"], false], ["get", "w"], 1],
    "propertySpec": {"type": "number"},
    "inputs": [[{}, {"properties": {"w": 3}}], [{}, {"properties": {"w": 3}, "featureState": {"hover": true}}],
               [{}, {"properties": {"w": 3}, "featureState": {"hover": false}}]]},
  "probe/state-color": {"expression": ["coalesce", ["feature-state", "color"], ["get", "color"], "black"],
    "propertySpec": {"type": "color"},
    "inputs": [[{}, {"properties": {"color": "red"}}], [{}, {"properties": {"color": "red"}, "featureState": {"color": "blue"}}],
               [{}, {"properties": {}}]]}
}"#;

fn prepare() -> (Vec<u32>, Vec<Case>, Counts) {
    let mut fixtures = json::parse(FIXTURES).expect("fixture bundle parses");
    let probes = json::parse(PROBES).expect("probes parse");
    if let (json::JsonValue::Object(all), json::JsonValue::Object(probes)) =
        (&mut fixtures.value, probes.value)
    {
        all.extend(probes);
    }
    let mut props = Vec::new();
    let mut cases = Vec::new();
    let mut counts = Counts::default();
    for member in fixtures.as_object().expect("bundle is an object") {
        counts.fixtures += 1;
        let fixture = &member.value;
        let (spec, expected) = fixture_spec(fixture);
        let expression = fixture.get("expression").expect("expression");
        let Ok(parsed) = compile(expression, &spec, expected.as_ref()) else {
            continue;
        };
        counts.compiled += 1;
        let program = match Program::compile(parsed.expression()) {
            Ok(p) => p,
            Err(e) => {
                *counts.fallbacks.entry(e).or_default() += 1;
                continue;
            }
        };
        counts.evaluable += 1;
        let layout = Layout::new([&program]);
        let at = props.len() as u32;
        props.extend(layout.encode(&program, &Value::Null));
        let inputs = fixture
            .get("inputs")
            .and_then(Json::as_array)
            .unwrap_or_default();
        for (i, input) in inputs.iter().enumerate() {
            let pair = input.as_array().expect("input pair");
            let (globals, feature) = (&pair[0], &pair[1]);
            let json = json_feature(feature);
            let mut value = |source: &Source| match source {
                Source::Property(key) => json.get_by_name(key),
                Source::Id => Some(json.id()),
                Source::GeometryType => Some(
                    json.geometry_type()
                        .map_or(Value::Null, |t| Value::String(t.into())),
                ),
                Source::State(_) => unreachable!("state inputs pack per state"),
            };
            let columns = props.len() as u32;
            let mut words = vec![0; layout.column_words() as usize];
            layout.pack_feature(&mut value, &mut words);
            props.extend(words);
            let state_at = match feature.get("featureState") {
                Some(s) => {
                    let mut words = vec![0; layout.state_words() as usize];
                    layout.pack_state(&state(Some(s)), &mut words);
                    let at = props.len() as u32;
                    props.extend(words);
                    at
                }
                None => u32::MAX,
            };
            let z = zoom(globals);
            let bits = z.to_bits();
            cases.push(Case {
                fixture: member.key.clone(),
                input: i,
                want: evaluate(&parsed, fixture, input).map_err(|_| ()),
                row: [
                    at,
                    columns,
                    state_at,
                    (z as f32).to_bits(),
                    bits as u32,
                    (bits >> 32) as u32,
                    0,
                    0,
                ],
            });
        }
    }
    (props, cases, counts)
}

const HARNESS: &str = r"
@group(0) @binding(0) var<storage, read> props: array<u32>;
@group(0) @binding(1) var<storage, read> cases: array<array<u32, 8>>;
@group(0) @binding(2) var<storage, read_write> results: array<vec4<u32>>;

fn prop_at(i: u32) -> u32 { return props[i]; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&cases) {
        return;
    }
    let c = cases[id.x];
    let zoom = expr_zoom(bitcast<f32>(c[3]), vec2<u32>(c[4], c[5]));
    results[id.x] = expr_eval(c[0], c[1], c[2], zoom);
}
";

async fn dispatch(props: &[u32], rows: &[u32], count: usize) -> Result<Vec<u32>, String> {
    use wgpu::util::DeviceExt;
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .map_err(|e| format!("no adapter: {e}"))?;
    let info = adapter.get_info();
    report(&format!("GPU: {} ({:?})", info.name, info.backend));
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .map_err(|e| format!("no device: {e}"))?;
    let source = format!("{HARNESS}{}", orb_shaders::EXPRESSION_MODULE);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("expression equivalence"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("expression equivalence"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let storage = |label, words: &[u32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(words),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let props = storage("props", props);
    let cases = storage("cases", rows);
    let size = (count * 16) as u64;
    let results = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("results"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[&props, &cases, &results]
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(count.div_ceil(64) as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&results, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);
    let (send, receive) = futures_channel::oneshot::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = send.send(r);
    });
    #[cfg(not(target_arch = "wasm32"))]
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| e.to_string())?;
    receive
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let mapped = readback.get_mapped_range(..).map_err(|e| e.to_string())?;
    Ok(mapped
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect())
}

fn unorm16(w: u32) -> [f32; 2] {
    [(w & 0xffff) as f32 / 65535.0, (w >> 16) as f32 / 65535.0]
}

/// f32 output arithmetic against the VM's f64.
fn close(gpu: f64, vm: f64) -> bool {
    if vm.is_nan() || gpu.is_nan() {
        return vm.is_nan() && gpu.is_nan();
    }
    if vm.is_infinite() || gpu.is_infinite() {
        return vm == gpu || (vm as f32).is_infinite() && (vm as f32) == gpu as f32;
    }
    (gpu - vm).abs() <= 1e-4 * vm.abs().max(1.0)
}

fn agrees(got: &[u32], want: &Result<Value<'static>, ()>) -> bool {
    match want {
        Err(()) => got[0] == 5,
        Ok(Value::Null) => got[0] == 0,
        Ok(Value::Bool(b)) => got[0] == 2 && got[1] == u32::from(*b),
        Ok(Value::Number(n)) => {
            got[0] == 1
                && close(
                    f64::from_bits(u64::from(got[2]) << 32 | u64::from(got[1])),
                    *n,
                )
                && close(f64::from(f32::from_bits(got[3])), *n)
        }
        Ok(Value::Color(c)) => {
            let [r, g] = unorm16(got[1]);
            let [b, a] = unorm16(got[2]);
            got[0] == 4
                && [r, g, b, a]
                    .iter()
                    .zip(c.straight())
                    .all(|(x, y)| (x - y).abs() <= 1e-4)
        }
        Ok(_) => false,
    }
}

fn report(line: &str) {
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_test::console_log!("{line}");
    #[cfg(not(target_arch = "wasm32"))]
    println!("{line}");
}

async fn run() {
    let (props, cases, counts) = prepare();
    let rows: Vec<u32> = cases.iter().flat_map(|c| c.row).collect();
    let got = match dispatch(&props, &rows, cases.len()).await {
        Ok(got) => got,
        Err(e) => {
            report(&format!("GPU expression fixtures skipped: {e}"));
            return;
        }
    };
    let mut failures = Vec::new();
    let mut agreeing: BTreeMap<&str, bool> = BTreeMap::new();
    for (case, got) in cases.iter().zip(got.as_chunks::<4>().0) {
        let ok = agrees(got, &case.want);
        *agreeing.entry(&case.fixture).or_insert(true) &= ok;
        if !ok {
            failures.push(format!(
                "{} input {}: VM {:?}, GPU {got:?}",
                case.fixture, case.input, case.want
            ));
        }
    }
    let fixtures_agreeing = agreeing.values().filter(|&&ok| ok).count();
    report(&format!(
        "GPU expression fixtures (with {} precision probes): {} of {} compile on the CPU; {} GPU-evaluable, \
         {} fall back {:?}; {fixtures_agreeing}/{} evaluable fixtures and {}/{} inputs agree with the VM",
        counts.fixtures - 583,
        counts.compiled,
        counts.fixtures,
        counts.evaluable,
        counts.compiled - counts.evaluable,
        counts.fallbacks,
        agreeing.len(),
        cases.len() - failures.len(),
        cases.len(),
    ));
    assert!(
        failures.is_empty(),
        "{} inputs disagree:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
async fn gpu_interpreter_matches_the_vm() {
    run().await;
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_interpreter_matches_the_vm() {
    pollster::block_on(run());
}
