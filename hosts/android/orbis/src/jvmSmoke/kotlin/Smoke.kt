// The generated Kotlin's smoke test on the host's JVM (plan §14.1): the JNI
// shim links and loads, a map draws a style into a texture target, and
// errors throw with their message. The full binding scenario runs in Rust;
// the emitters' tests cover every API shape. `cargo xtask bindings --test`
// runs it.

import dev.orbis.*

val STYLE = """
    {"version": 8, "center": [0, 0], "zoom": 0,
     "sources": {"box": {"type": "geojson", "data": {
       "type": "Feature", "id": 7, "properties": {},
       "geometry": {"type": "Polygon", "coordinates":
         [[[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]]]}}}},
     "layers": [
       {"id": "background", "type": "background", "paint": {"background-color": "#ff0000"}},
       {"id": "box", "type": "fill", "source": "box", "paint": {"fill-color": "#0000ff"}}]}
""".trimIndent()

fun main() {
    check(OrbRuntime.abiVersion() == 5u)
    OrbRuntime.create(0u).use { runtime ->
        OrbTarget.createTexture(runtime, 64u, 64u, 1.0).use { target ->
            OrbMap.create(runtime, OrbFeel.Desktop, 0.0, 0.0, 0.0, 0.0).use { map ->
                val bad = runCatching { map.setStyleJson("{not json") }.exceptionOrNull()
                check(bad is OrbErrorException && bad.error == OrbError.InvalidArgument && !bad.message.isNullOrEmpty())

                // Draws until the blue box shows in the middle.
                map.setStyleJson(STYLE)
                val started = System.nanoTime()
                val i = (32 * 64 + 32) * 4
                while (true) {
                    map.render(target, 0.0)
                    target.readPixels()
                    val pixels = target.pixels()
                    if ((i until i + 4).map { pixels[it].toInt() and 0xff } == listOf(0, 0, 255, 255)) break
                    check(System.nanoTime() - started < 30_000_000_000) { "the box never showed" }
                    Thread.sleep(2)
                }
                map.hitTest(32.0, 32.0)!!.use { hit -> check(hit.layer() == "box" && hit.id() == "7") }
            }
        }
    }
    println("Smoke.kt: ok")
}
