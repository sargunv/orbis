// Opens a demo page in a desktop browser no driver controls (Safari,
// Firefox, a user's Chrome) and prints the result the page posts back to
// `?report=` (pipelines.html, measure.html with `?auto`).
//
//   node scripts/open.mjs <app> <page> [query]   e.g. Safari pipelines.html
import { execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { serveDemo } from "./serve.mjs";

const [app, page, query = ""] = process.argv.slice(2);
const server = await serveDemo();
const result = await new Promise((resolve, reject) => {
  const receiver = createServer((req, res) => {
    let body = "";
    req.on("data", (c) => { body += c; });
    req.on("end", () => { res.writeHead(204).end(); receiver.close(); resolve(body); });
  });
  receiver.listen(0, "127.0.0.1", () => {
    const report = `http://127.0.0.1:${receiver.address().port}/`;
    const url = `${server.url}${page}?${query}${query ? "&" : ""}report=${encodeURIComponent(report)}`;
    execFileSync("open", ["-a", app, url]);
  });
  setTimeout(() => reject(new Error("no result in 2 minutes")), 120_000);
});
server.close();
console.log(result);
