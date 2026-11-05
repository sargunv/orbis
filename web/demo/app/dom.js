// A small element builder: h("button", { class: "tool", onclick }, "Text").
export function h(tag, props = {}, ...children) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key.startsWith("on")) element.addEventListener(key.slice(2), value);
    else if (key === "class") element.className = value;
    else if (key === "style") Object.assign(element.style, value);
    else if (key === "value") element.value = value;
    else if (key in element && typeof value !== "string") element[key] = value;
    else element.setAttribute(key, value === true ? "" : value);
  }
  element.append(...children.flat().filter((c) => c !== undefined && c !== null && c !== false));
  return element;
}

/** An icon from app.css's set. */
export const icon = (name) => h("span", { class: `icon icon-${name}`, "aria-hidden": "true" });

/** Text for assistive technology only. */
export const hidden = (text) => h("span", { class: "visually-hidden" }, text);

/**
 * Attribution HTML from a style or TileJSON, reduced to text and http(s)
 * links: sources are third-party.
 */
export function safeHtml(html) {
  const doc = new DOMParser().parseFromString(`<body>${html}</body>`, "text/html");
  const out = document.createDocumentFragment();
  const walk = (node, into) => {
    for (const child of node.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) into.append(child.textContent);
      else if (child.nodeName === "A" && /^https?:/.test(child.getAttribute("href") ?? "")) {
        const link = h("a", { href: child.getAttribute("href"), target: "_blank", rel: "noopener" });
        walk(child, link);
        into.append(link);
      } else if (child.nodeType === Node.ELEMENT_NODE) walk(child, into);
    }
  };
  walk(doc.body, out);
  return out;
}
