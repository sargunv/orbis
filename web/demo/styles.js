import mvt from "./data/tokyo-mvt.json?url";
import mlt from "./data/tokyo-mlt.json?url";

import playground from "./data/playground.json?url";
import scripts from "./data/scripts.json?url";

export function demoStyle(params) {
  const format = params.get("format") ?? "mvt";
  if (!["mvt", "mlt"].includes(format)) throw new Error("format must be mvt or mlt");
  // `style=scripts`: the Cairo, Delhi and Bangkok page's style.
  const style = params.get("style") === "scripts" ? scripts : params.get("style");
  return new URL(style ?? (location.pathname.endsWith("playground.html") ? playground : { mvt, mlt }[format]), location.href).href;
}
