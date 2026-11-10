// The service worker that makes a page cross-origin isolated when it
// compares with an engine whose threads need shared memory (MapLibre Native,
// app/compare.js), on a host that sets no headers. It isolates only
// navigations whose `?compare=` names one of the engines in its own URL's
// `?engines=`, so every other page loads as it would without it. Its COEP
// is `credentialless`, so cross-origin tiles, fonts and sprites without
// CORP headers still load. app/isolation.js registers it.
const engines = new Set(
  new URL(location.href).searchParams.get("engines")?.split(",") ?? [],
);
const wanted = (url) =>
  (new URL(url).searchParams.get("compare") ?? "")
    .split(",")
    .some((e) => engines.has(e));

addEventListener("install", (event) => {
  skipWaiting();
  // Where the browser can route requests itself, only navigations and
  // worker scripts wake this worker; the rest go straight to the network.
  const routes = event.addRoutes?.([
    { condition: { requestMode: "navigate" }, source: "fetch-event" },
    { condition: { requestDestination: "worker" }, source: "fetch-event" },
    { condition: { urlPattern: new URLPattern({}) }, source: "network" },
  ]);
  if (routes) event.waitUntil(routes.catch(() => {}));
});

addEventListener("fetch", (event) => {
  const { request } = event;
  // An isolated page's workers need the policy too, from any page: one may
  // have turned the engine off since it loaded and still be isolated.
  const isolate =
    request.mode === "navigate"
      ? wanted(request.url)
      : request.destination === "worker" &&
        new URL(request.url).origin === location.origin;
  if (isolate) event.respondWith(isolated(request));
});

async function isolated(request) {
  const response = await fetch(request);
  if (response.type === "opaque" || response.type === "opaqueredirect")
    return response;
  const headers = new Headers(response.headers);
  headers.set("cross-origin-opener-policy", "same-origin");
  headers.set("cross-origin-embedder-policy", "credentialless");
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}
