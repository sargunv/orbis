# MapLibre style-spec expressions

The complete `test/integration/expression/tests/**/test.json` suite from
[maplibre-style-spec](https://github.com/maplibre/maplibre-style-spec), bundled
by fixture path into `expression-fixtures.json` with no filtering or changes.

- Version: 26.5.1
- Commit: c25cc5da789337fca2df8d0eaaeeecd5eea7f634 (tag v26.5.1)
- Version source: `package-lock.json` of GL JS commit
  2b68611a4e2d4f7f076f6c6ebf758e2d53f91471 (pinned in `xtask/src/fixtures.rs`).
- Fixtures: **583**.
- License: upstream `LICENSE.txt`, copied as `LICENSE-style-spec.txt`.
- Regenerate: `cargo xtask fixtures expressions`.

Native and wasm conformance tests embed the same JSON, so neither needs
filesystem I/O, serde or a separate download.

Run natively with `cargo test -p orb-expr`. Run in headless Chrome with:

```sh
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
  cargo test -p orb-expr --lib --target wasm32-unknown-unknown -- --nocapture
```

`wasm-bindgen-test-runner` uses ChromeDriver on PATH, or `CHROMEDRIVER`.
For a nonstandard Chrome installation, set `WASM_BINDGEN_TEST_WEBDRIVER_JSON`
to a WebDriver capabilities JSON file containing
`{"goog:chromeOptions":{"binary":"/path/to/chrome"}}`.
