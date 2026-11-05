// Frame scheduling: frames run on the display's refresh
// (requestAnimationFrame) and only while something asks for them: no
// frames when idle. Each frame gets its animation frame's timestamp, on
// the clock input events' `timeStamp`s use.

/**
 * Runs `frame` on the next display frame after each `request()`, and again
 * while it returns true. `stop()` ends it for good.
 */
export function frameLoop(frame: (now: number) => boolean, onError: (e: unknown) => void) {
  let scheduled = false;
  let stopped = false;
  const request = () => {
    if (scheduled || stopped) return;
    scheduled = true;
    requestAnimationFrame((now) => {
      scheduled = false;
      if (stopped) return;
      try {
        if (frame(now)) request();
      } catch (e) {
        onError(e);
      }
    });
  };
  return { request, stop: () => { stopped = true; }, get scheduled() { return scheduled; } };
}
