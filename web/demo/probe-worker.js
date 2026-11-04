import { adapterInfo } from "./gpu-probe.js";

self.onmessage = async () => self.postMessage({
  offscreenCanvas: typeof OffscreenCanvas !== "undefined",
  ...(await adapterInfo(self.navigator.gpu)),
});
