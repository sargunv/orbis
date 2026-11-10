// Cross-origin isolation, on demand: an engine whose threads share memory
// (MapLibre Native) needs it, and Orbis deliberately doesn't. The demo's
// host sets no headers, so ../isolate.js, a service worker, adds them to
// the navigations that compare with such an engine; this registers it and
// reloads into isolation. Pages that don't name such an engine load as if
// it weren't there.

const TRIED = "orbis-isolation-tried";

/**
 * Reloads the page cross-origin isolated for `engines`, the comparison's
 * engines that need it, once its URL names one. Throws, saying why, where
 * the browser won't, rather than reloading again.
 */
export async function isolate(engines, { beforeReload } = {}) {
  const tried = read();
  write(null);
  if (tried !== null && Date.now() - tried < 60_000) {
    throw new Error(
      "this browser didn't isolate the page, which its threads need",
    );
  }
  if (!("serviceWorker" in navigator))
    throw new Error(
      "it needs service workers, which this browser doesn't have",
    );
  const base = import.meta.env.BASE_URL;
  await navigator.serviceWorker.register(
    `${base}isolate.js?engines=${engines.join(",")}`,
    { scope: base },
  );
  await navigator.serviceWorker.ready;
  // Without a note of this attempt, a browser that won't isolate would reload forever.
  if (!write(Date.now()))
    throw new Error("this browser blocks the session storage its reload needs");
  beforeReload?.();
  location.reload();
  // The page goes; nothing after this runs.
  await new Promise(() => {});
}

/** Forgets an attempt that worked. */
export function isolated() {
  if (crossOriginIsolated) write(null);
  return crossOriginIsolated;
}

function read() {
  try {
    const at = sessionStorage.getItem(TRIED);
    return at === null ? null : Number(at);
  } catch {
    return null;
  }
}
function write(value) {
  try {
    if (value === null) sessionStorage.removeItem(TRIED);
    else sessionStorage.setItem(TRIED, String(value));
    return true;
  } catch {
    return false;
  }
}
