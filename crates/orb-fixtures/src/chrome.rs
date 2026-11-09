//! The Chrome driver (`web/scripts/render-fixtures.mjs`): one Chromium
//! process drawing fixtures through the `orbis` package; selection and
//! comparison stay in Rust.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

use crate::{images::Image, metadata::Metadata};

pub(crate) struct Chrome {
    child: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
    pub(crate) version: Value,
    pub(crate) max_size: u32,
}

impl Chrome {
    /// The driver, drawing on WebGPU.
    pub(crate) fn new(root: &Path) -> Result<Self, String> {
        let mut child = Command::new("node")
            .arg(root.join("web/scripts/render-fixtures.mjs"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| {
                format!("start Chrome driver: {e}; run `mise install` and `pnpm -C web install`")
            })?;
        let input = child.stdin.take().expect("piped stdin");
        let output = BufReader::new(child.stdout.take().expect("piped stdout"));
        let mut chrome = Self {
            child,
            input: Some(input),
            output,
            version: Value::Null,
            max_size: 0,
        };
        let ready = chrome.receive()?;
        if ready["ready"] != true {
            return Err(format!("Chrome initialization: {ready}"));
        }
        chrome.version = ready["browser"].clone();
        chrome.max_size = ready["max_size"]
            .as_u64()
            .ok_or("missing Chrome device limit")? as u32;
        Ok(chrome)
    }

    fn receive(&mut self) -> Result<Value, String> {
        let mut line = String::new();
        self.output
            .read_line(&mut line)
            .map_err(|e| e.to_string())?;
        serde_json::from_str(&line).map_err(|e| format!("Chrome driver response: {e}: {line}"))
    }

    /// Sends a fixture's request, with `extra` fields, and reads the reply.
    fn send(&mut self, style: &str, metadata: &Metadata, extra: Value) -> Result<Value, String> {
        let (width, height) = metadata.physical_size();
        let mut request = json!({"style":style, "width":width, "height":height,
            "ratio":metadata.pixel_ratio,"timeout_ms":metadata.timeout_ms,
            "fade_duration":metadata.fade_duration,
            "cross_source_collisions":metadata.cross_source_collisions,
            "local_ideographs":metadata.local_ideographs,
            "max_pitch":metadata.max_pitch,
            "terrain_skirts":metadata.terrain_skirts,
            "operations":metadata.operations});
        request
            .as_object_mut()
            .expect("an object")
            .extend(extra.as_object().expect("an object").clone());
        let input = self.input.as_mut().expect("open transport");
        serde_json::to_writer(&mut *input, &request).map_err(|e| e.to_string())?;
        input
            .write_all(b"\n")
            .and_then(|_| input.flush())
            .map_err(|e| e.to_string())?;
        self.receive()
    }

    /// Renders a test: its RGBA pixels, unpremultiplied as the native
    /// readback is, or `None` with the driver's error in the response.
    pub(crate) fn render(
        &mut self,
        style: &str,
        metadata: &Metadata,
        output: &Path,
    ) -> Result<(Option<Image>, Value), String> {
        let response = self.send(style, metadata, json!({"output": output}))?;
        if !response["error"].is_null() {
            return Ok((None, response));
        }
        let (width, height) = metadata.physical_size();
        let pixels = std::fs::read(output).map_err(|e| e.to_string())?;
        std::fs::remove_file(output).map_err(|e| e.to_string())?;
        if pixels.len() != width as usize * height as usize * 4 {
            return Err("Chrome returned the wrong number of RGBA bytes".into());
        }
        let mut image = Image {
            width,
            height,
            pixels,
        };
        image.unpremultiply();
        Ok((Some(image), response))
    }

    /// Settles a query test's style as [`render`](Self::render) does, then
    /// queries the box `[x1, y1, x2, y2]`; returns the features.
    pub(crate) fn query(
        &mut self,
        style: &str,
        metadata: &Metadata,
        query: [f64; 4],
    ) -> Result<Value, String> {
        let response = self.send(style, metadata, json!({"query": query}))?;
        if !response["error"].is_null() {
            return Err(format!("Chrome query: {}", response["error"]));
        }
        let features = response["features"]
            .as_str()
            .ok_or("Chrome query: no features")?;
        serde_json::from_str(features).map_err(|e| format!("Chrome query JSON: {e}"))
    }
}

impl Drop for Chrome {
    fn drop(&mut self) {
        // Also reap on report/I/O errors; never leave a browser driver behind.
        self.input.take();
        let _ = self.child.wait();
    }
}
