// The debug overlay: the last frame's counters and the build. Off by default; `?debug` or the menu turns it on.
import { h } from "./dom.js";

const kib = (bytes) => `${(bytes / 1024).toFixed(1)} KiB`;
const COMMIT = __ORBIS_COMMIT__;
const mib = (bytes) => `${(bytes / 1048576).toFixed(1)} MiB`;

/** GPU memory allocated, the scene's growable pools
 * (used / allocated / ceiling) and the tile cache. */
function memoryRows(m) {
  if (!m?.allocated) return [];
  const pool = (used, allocated, ceiling) =>
    `${mib(used)} / ${mib(allocated)} / ${mib(ceiling)}`;
  return [
    ["GPU memory", mib(m.allocated)],
    ["vertices", pool(m.verticesUsed, m.verticesAllocated, m.verticesCeiling)],
    ["indices", pool(m.indicesUsed, m.indicesAllocated, m.indicesCeiling)],
    ["compacted", `${mib(m.compactedUsed)} / ${mib(m.compactedAllocated)}`],
    ...(m.cpuCopies ? [["CPU copies", mib(m.cpuCopies)]] : []),
    ["textures", mib(m.textures)],
    ["cache", `${m.cachedTiles} of ${m.cacheLimit} tiles, ${mib(m.cached)}`],
  ];
}

export function hud() {
  const root = document.getElementById("hud");
  const list = h("dl");
  const commit = h(
    "a",
    { href: `https://github.com/sargunv/orbis/commit/${COMMIT.sha}` },
    `${COMMIT.short}${COMMIT.dirty ? "-dirty" : ""}`,
  );
  root.replaceChildren(
    list,
    h("div", { class: "hud-foot" }, h("span", {}, "Build ", commit)),
  );
  let latest;
  let hovered = "";
  let recovery = "";
  let queued = false;

  const draw = () => {
    queued = false;
    const ack = latest;
    if (!ack || root.hidden) return;
    const rows = [
      ["zoom", ack.zoom.toFixed(2)],
      ["CPU", `${ack.ms.toFixed(1)} ms`],
      ["draws", ack.draws],
      ["dispatches", ack.dispatches],
      ["candidates", ack.candidates],
      [
        "tiles",
        ack.tilesLoading
          ? `${ack.tilesDrawn}, ${ack.tilesLoading} loading`
          : ack.tilesDrawn,
      ],
      ["bundle", ack.bundleRecorded ? "recorded" : "replayed"],
      ...(ack.pipelinesPending || ack.layersWaiting
        ? [
            [
              "pipelines",
              `${ack.pipelinesPending} compiling, ${ack.layersWaiting} layers waiting`,
            ],
          ]
        : []),
      ["placements", ack.placementsWritten],
      ["paints", ack.paintsWritten],
      ["from JS", kib(ack.uploadedFromJs)],
      ["from wasm", kib(ack.uploadedFromWasm)],
      ...(ack.texelsFromJs || ack.texelsFromWasm
        ? [
            ["texels JS", kib(ack.texelsFromJs)],
            ["texels wasm", kib(ack.texelsFromWasm)],
          ]
        : []),
      ...(ack.jobsDone
        ? [
            ["jobs", ack.jobsDone],
            ["headers", kib(ack.headerBytes)],
          ]
        : []),
      ...memoryRows(ack.memory),
      ["generation", ack.generation],
      ["state", ack.idle ? "idle" : "drawing"],
      ...(recovery ? [["recovered", recovery]] : []),
      ...(hovered ? [["hover", hovered]] : []),
    ];
    list.replaceChildren(
      ...rows.map(([key, value]) =>
        h("div", {}, h("dt", {}, key), h("dd", {}, String(value))),
      ),
    );
  };
  const request = () => {
    if (!queued) {
      queued = true;
      requestAnimationFrame(draw);
    }
  };

  return {
    get visible() {
      return !root.hidden;
    },
    set visible(on) {
      root.hidden = !on;
      request();
    },
    frame(ack) {
      latest = ack;
      request();
    },
    hover(text) {
      hovered = text;
      request();
    },
    recovered(text) {
      recovery = text;
      request();
    },
  };
}
