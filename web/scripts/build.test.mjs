import assert from "node:assert/strict";
import {
  appendFile,
  cp,
  mkdir,
  mkdtemp,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { build } from "vite";

import config from "../vite.config.js";

test("built styles pin their archives, and archive changes propagate to entry JS", async () => {
  const dir = await mkdtemp(join(tmpdir(), "orbis-vite-"));
  try {
    const root = join(dir, "demo");
    await mkdir(root);
    await cp(join(config.root, "data"), join(root, "data"), {
      recursive: true,
    });
    await cp(join(config.root, "styles.js"), join(root, "styles.js"));
    // The map options, which open a restyle as the app does.
    await cp(join(config.root, "app"), join(root, "app"), { recursive: true });
    await writeFile(
      join(root, "index.html"),
      `<!doctype html><script type="module">
      import { demoStyle } from "./styles.js";
      document.body.textContent = demoStyle(new URLSearchParams());
    </script>`,
    );
    const bundle = async () =>
      (
        await build({
          ...config,
          configFile: false,
          logLevel: "silent",
          root,
          build: {
            ...config.build,
            write: false,
            rolldownOptions: {
              input: join(root, "index.html"),
            },
          },
        })
      ).output;
    const before = await bundle();
    // Glyphs and sprites keep their names: styles fetch them by template.
    // The glyphs' font licence ships with them. The isolating service
    // worker and MapLibre Native's browser module keep theirs too.
    for (const entry of before) {
      assert.match(
        entry.fileName,
        /^index\.html$|^isolate\.js$|^native\/[\d.]+\/webgl\/(?:maplibre_native_ffi\.(?:mjs|wasm)|LICENSE|licenses\/[\w.-]+)$|^assets\/.+-[\w-]{8}\.(?:js|json|wasm|pmtiles|bin|glb)$|^data\/(?:glyphs\/(?:[^/]+\/\d+-\d+\.pbf|OFL\.txt)|sprites\/ofm(?:@2x)?\.(?:json|png))$/,
      );
    }
    const find = (output, prefix) => {
      const entry = output.find((entry) =>
        entry.fileName.startsWith(`assets/${prefix}-`),
      );
      assert.ok(entry, `missing ${prefix}`);
      return entry;
    };
    const style = (output) =>
      output.find((entry) =>
        /^assets\/tokyo-mvt-.*\.json$/.test(entry.fileName),
      );
    const archive = (output) =>
      output.find((entry) =>
        /^assets\/tokyo-mvt-.*\.pmtiles$/.test(entry.fileName),
      );
    assert.equal(
      JSON.parse(style(before).source).sources.openmaptiles.url,
      `pmtiles://${config.base}${archive(before).fileName}`,
    );
    const { glyphs, sprite } = JSON.parse(style(before).source);
    assert.equal(glyphs, `${config.base}data/glyphs/{fontstack}/{range}.pbf`);
    assert.ok(
      before.some(
        (entry) => `${config.base}${entry.fileName}` === `${sprite}.json`,
      ),
      "sprite emitted",
    );

    // A valid PMTiles header alone cannot establish content identity. Even
    // a change after the header must invalidate the archive, style and JS.
    await appendFile(join(root, "data/tokyo-mvt.pmtiles"), new Uint8Array([0]));
    const after = await bundle();
    assert.notEqual(archive(before).fileName, archive(after).fileName);
    assert.notEqual(style(before).fileName, style(after).fileName);
    assert.notEqual(
      find(before, "index").fileName,
      find(after, "index").fileName,
    );
    assert.equal(
      JSON.parse(style(after).source).sources.openmaptiles.url,
      `pmtiles://${config.base}${archive(after).fileName}`,
    );
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
