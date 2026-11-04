# 0005: Expression locale service and conformance

Scope: M1 expression CPU evaluator, plan §11.3.

## Outcome

`orb-expr` implements the 89 public operators in style-spec 26.5.1, including
its newer `semiliteral` operator and property value types. Parsing, post-order
IR and CPU evaluation remain the same code on native and wasm. Register,
column and GPU backends remain later M1 work.

`Locale` belongs in `orb-expr`, the core crate that consumes it. Its narrow
interface covers search comparison, resolved locale, number formatting and
Unicode case mapping. `orb-host` can provide platform implementations later
without an expression-to-host dependency. `parse_with_locale` uses the host
service during constant folding; `EvaluationContext::locale` uses it during
feature evaluation. A host should supply the same service for both.

The deterministic built-in service has an English (`en-US`) default, Latin
search collation with German and Swedish tailoring, English/German/French
number separators, common currencies and units, and Rust's Unicode case
mapping. It passes the pinned suite's Node-default locale cases. It is not a
complete Unicode collation or CLDR implementation: other scripts, locale
negotiation, numbering systems and the full ECMA-402 currency/unit repertoire
need a platform service. This keeps a large locale data payload out of the
initial wasm module. No JS VM, serde or serde_json is added to core.

## One fixture source and one comparison implementation

GL JS commit `2b68611a4e2d4f7f076f6c6ebf758e2d53f91471` locks style-spec to
26.5.1, tag commit `c25cc5da789337fca2df8d0eaaeeecd5eea7f634`. This version has
583 expression fixtures, six more than the plan's original count. All are
bundled, without filtering, in `crates/orb-expr/tests/data/`; `cargo xtask
fixtures expressions` regenerates the bundle and upstream license.

A bundle is the single source for both targets and needs no filesystem on
wasm. `src/conformance.rs` validates compilation types and dependency flags,
exact diagnostics and error paths, and all outputs at the reference runner's
six-significant-figure precision. Geometry inputs use the reference runner's
tile quantization before evaluation. Future evaluators can reuse these cases.

The native entry is an ordinary Rust test. The wasm entry uses
`wasm-bindgen-test` in headless Chrome, with the same Rust test body. The
renderer fixture transport is unnecessary here: expressions require no GPU,
map or HTTP asset setup. The browser test dependency and fixture JSON are
compiled only for tests and do not enter `core.wasm`.

## Measurements

- Native CPU expression fixtures: **583/583**.
- Chrome 157 CPU expression fixtures, compiled to wasm: **583/583**.
- `cargo xtask web`: **359 KiB brotli**, versus **336 KiB** on the same
  tree without this implementation (**+23 KiB**), **24 KiB** over the
  **335 KiB** first measured before the expression work. The gate is
  **450 KiB**. Raw wasm: **1103 KiB** (baseline **1023 KiB**).

The evaluator accepts feature/global state, tile-local geometry, line progress,
heatmap density, elevation, accumulated values, image availability and script
support through `EvaluationContext`. Existing calls keep their default-input
behavior. Literal feature-property keys still resolve to columns at bind time;
computed property names retain their reference-defined dynamic lookup.

Native rendering preserves all original **99/101** results and adds two passes
for previously unsupported HCL/Lab background-color fixtures: **101/103** in
scope, with 319 skips after the parallel mirror expansion to 422 fixtures. The existing failures are
`fill-outline-color/multiply` and `line-miter-limit/default`.
