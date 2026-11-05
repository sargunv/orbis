// Engine-drawn markers and the location layer (orbis's batteries): with
// "Drop pins" on, a click drops a pin and a click on a pin lifts it; "My
// location" follows the browser's geolocation.
/** `map()` returns the current map, or undefined; `say` shows a toast. */
export function batteries({ map, say }) {
  const pins = document.getElementById("pins-toggle");
  const location = document.getElementById("location-toggle");
  let dropping = false;
  let watch;

  pins.addEventListener("click", () => {
    dropping = !dropping;
    pins.setAttribute("aria-checked", String(dropping));
    if (dropping) say("Click the map to drop a pin, and a pin to lift it.");
  });

  location.addEventListener("click", () => {
    if (watch !== undefined) {
      navigator.geolocation.clearWatch(watch);
      watch = undefined;
      location.setAttribute("aria-checked", "false");
      void map()?.setLocation(null);
      return;
    }
    if (!navigator.geolocation) return say("This browser has no geolocation.", { error: true });
    location.setAttribute("aria-checked", "true");
    let first = true;
    watch = navigator.geolocation.watchPosition(({ coords }) => {
      const current = map();
      if (!current) return;
      const lngLat = [coords.longitude, coords.latitude];
      void current.setLocation({ lngLat, accuracy: coords.accuracy, heading: coords.heading ?? undefined });
      if (first) {
        first = false;
        void current.camera.flyTo({ center: lngLat, zoom: Math.max(current.camera.current.zoom, 15) });
      }
    }, (e) => {
      say(`Location unavailable: ${e.message}`, { error: true });
      location.setAttribute("aria-checked", "false");
      watch = undefined;
    }, { enableHighAccuracy: true });
  });

  return {
    /** Handles a click while dropping pins; whether it did. */
    async click(point, lngLat) {
      const current = map();
      if (!dropping || !current || !lngLat) return false;
      const [pin] = await current.queryRenderedFeatures(point, { layers: ["orbis-markers"] });
      if (pin) await current.removeMarker(Number(pin.id));
      else await current.addMarker({ lngLat });
      return true;
    },
  };
}

