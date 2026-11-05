// Offline regions in headless Chromium: opens the live map, downloads the
// view from the Offline areas dialog, then reloads with every request but
// the demo server's failing and checks the map draws without errors.
//
//   pnpm offline ['#zoom/lat/lon']
import { launchChromium, serveDemo } from "./serve.mjs";

const hash = process.argv[2] ?? "#13/35.6830/139.7520";
const server = await serveDemo();
const browser = await launchChromium();
try {
  const page = await browser.newPage({ viewport: { width: 800, height: 600 } });
  const errors = [];
  let offline = false;
  page.on("console", (m) => {
    if (m.type() !== "error") return;
    // The browser logs each request the cut network refuses.
    if (offline && m.text().includes("ERR_INTERNET_DISCONNECTED")) return;
    if (offline) errors.push(m.text());
  });
  const idle = () => page.waitForFunction(() => ["idle", "error"].includes(document.getElementById("app")?.dataset.map),
    null, { timeout: 60_000 });
  await page.goto(`${server.url}?style=liberty${hash}`);
  await idle();
  await page.click("#menu-button");
  await page.click("#offline-button");
  const started = Date.now();
  await page.click("#offline .primary");
  await page.waitForFunction(() => /Downloaded|failed/.test(document.querySelector(".offline-status")?.textContent ?? ""),
    null, { timeout: 180_000 });
  const status = await page.textContent(".offline-status");
  console.log(`${status} (${Date.now() - started} ms)`);
  if (!status.startsWith("Downloaded")) throw new Error(status);

  offline = true;
  await page.route((u) => !u.href.startsWith(server.url), (route) => route.abort("internetdisconnected"));
  const reopened = Date.now();
  await page.reload();
  await idle();
  const state = await page.evaluate(() => document.getElementById("app").dataset.map);
  console.log(`reopened offline: ${state} in ${Date.now() - reopened} ms`);
  if (state !== "idle" || errors.length) throw new Error(`offline: ${state}\n${errors.join("\n")}`);
} finally {
  await browser.close();
  server.close();
}
