// Draws a map's frames only while it is woken, on the next animation frame
// (or a timer, in workers without requestAnimationFrame), and reports each
// as a frame ack: plain data that can cross to the page.
const nextFrame = globalThis.requestAnimationFrame?.bind(globalThis)
  ?? ((draw) => setTimeout(draw, 16));

export function frameLoop(onFrame, onError) {
  let map = null;
  let scheduled = false;
  const request = () => {
    if (scheduled || !map) return;
    scheduled = true;
    nextFrame(() => {
      scheduled = false;
      try {
        const started = performance.now();
        const info = map.render();
        const ms = performance.now() - started;
        const idle = info.idle;
        onFrame(ack(info, ms));
        info.free();
        if (!idle) request();
      } catch (e) {
        onError(e);
      }
    });
  };
  return { request, start(m) { map = m; request(); } };
}

function ack(info, ms) {
  return {
    ms,
    hitSnapshot: info.hit_snapshot,
    zoom: info.zoom,
    generation: info.generation,
    meshesUploaded: info.meshes_uploaded,
    rowsUploaded: info.rows_uploaded,
    draws: info.draws,
    dispatches: info.dispatches,
    candidates: info.candidates,
    tilesDrawn: info.tiles_drawn,
    tilesLoading: info.tiles_loading,
    bundleRecorded: info.bundle_recorded,
    placementsWritten: info.placements_written,
    paintsWritten: info.paints_written,
    idle: info.idle,
    warnings: info.warnings,
    uploadedFromWasm: info.uploaded_from_wasm,
    uploadedFromJs: info.uploaded_from_js,
    texelsFromWasm: info.texels_from_wasm,
    texelsFromJs: info.texels_from_js,
    headerBytes: info.header_bytes,
    jobsDone: info.jobs_done,
  };
}
