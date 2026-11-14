# MapLibre style-spec expressions

The complete `test/integration/expression/tests/**/test.json` suite from
[maplibre-style-spec](https://github.com/maplibre/maplibre-style-spec), bundled
by fixture path into `expression-fixtures.json` with no filtering or changes.

- Commit: 5d212533ee7456c1006b3c8d44e1f9e1d83b796c (main after v26.7.0, with
  `oklab()` and `interpolate-oklab`), ahead of the version the pinned GL JS
  locks, so that the suite covers what the spec approves as soon as it does.
- Fixtures: **601**.
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
