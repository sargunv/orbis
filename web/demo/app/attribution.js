// The attribution control: the active style's sources' attributions (from
// the style, or from TileJSON for sources that only link one), on one line
// that expands to all of them.
import { h, safeHtml } from "./dom.js";

const tileJson = new Map();
function fetchAttribution(url) {
  if (!tileJson.has(url)) {
    tileJson.set(
      url,
      fetch(url)
        .then((r) => (r.ok ? r.json() : {}))
        .then(
          (j) => j.attribution ?? "",
          () => "",
        ),
    );
  }
  return tileJson.get(url);
}

export function attribution() {
  const root = document.getElementById("attribution");
  let open = false;
  let generation = 0;
  const text = h("div", { class: "attribution-text", id: "attribution-text" });
  const toggle = h(
    "button",
    {
      class: "attribution-toggle",
      type: "button",
      "aria-expanded": "false",
      "aria-controls": "attribution-text",
      title: "Map data sources",
      onclick: () => {
        open = !open;
        root.classList.toggle("open", open);
        toggle.setAttribute("aria-expanded", String(open));
      },
    },
    h("span", { "aria-hidden": "true" }, "i"),
    h("span", { class: "visually-hidden" }, "Map data sources"),
  );
  root.replaceChildren(text, toggle);

  return {
    async update(style) {
      const mine = ++generation;
      const entries = await Promise.all(
        Object.values(style?.sources ?? {}).map((source) => {
          if (source.attribution) return source.attribution;
          if (typeof source.url === "string" && /^https?:/.test(source.url))
            return fetchAttribution(source.url);
          return "";
        }),
      );
      if (mine !== generation) return;
      const seen = new Set();
      const parts = [];
      for (const entry of entries) {
        const fragment = safeHtml(entry);
        const key = fragment.textContent.trim();
        if (!key || seen.has(key)) continue;
        seen.add(key);
        parts.push(h("span", {}, fragment));
      }
      text.replaceChildren(
        ...parts.flatMap((part, i) => (i ? [" ", part] : [part])),
      );
      root.hidden = parts.length === 0;
    },
  };
}
