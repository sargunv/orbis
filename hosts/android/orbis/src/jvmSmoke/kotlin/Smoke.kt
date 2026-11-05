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
    check(OrbRuntime.abiVersion() == 9u)
    OrbRuntime.create(0u).use { runtime ->
        OrbTarget.createTexture(runtime, 64u, 64u, 1.0).use { target ->
            OrbMap.create(runtime, OrbFeel.Desktop, 0.0, 0.0, 0.0, 0.0).use { map ->
                // The viewport, before a target draws.
                map.setViewport(64.0, 64.0)
                check(map.project(0.0, 0.0).let { Math.abs(it.x - 32.0) < 1e-6 && Math.abs(it.y - 32.0) < 1e-6 })
                val bad = runCatching { map.setStyleJson("{not json") }.exceptionOrNull()
                check(bad is OrbErrorException && bad.error == OrbError.InvalidArgument && !bad.message.isNullOrEmpty())

                // Draws until the blue box shows in the middle.
                // From a style a request rule finds on disk (files read offline).
                val dir = kotlin.io.path.createTempDirectory("orb-smoke").toFile()
                java.io.File(dir, "style.json").writeText(STYLE)
                check(runCatching { runtime.setRequestRules("{bad") }.isFailure)
                runtime.setRequestRules("""{"rewrites": [["smoke://", "file://${dir.absolutePath}/"]]}""")
                runtime.setOnline(false)
                check(!runtime.online())
                map.loadStyleUrl("smoke://style.json")
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

                // The style reads back.
                check(map.layerIds() == """["background","box"]""") { map.layerIds() }
                check(map.layer("box").contains("#0000ff"))
                check(runCatching { map.layer("nope") }.isFailure)

                // Edits built one by one apply as one generation; a value
                // that isn't JSON throws at once.
                OrbTransaction.create().use { tx ->
                    tx.setPaintProperty("box", "fill-color", "\"#00ff00\"")
                    tx.setPaintProperty("box", "fill-opacity", "1")
                    check(runCatching { tx.setFilter("box", "[==") }.isFailure)
                    val before = map.generation()
                    check(map.edit(tx) == before + 1u)
                }

                // Padding moves the centre down; a camera's projection
                // answers before any move; the eye round-trips.
                val nan = Double.NaN
                val keep = OrbPadding(nan, nan, nan, nan)
                map.jumpTo(OrbCameraTarget(nan, nan, nan, nan, nan, nan, nan, nan, keep.copy(top = 32.0)))
                val centre = map.project(0.0, 0.0)
                check(centre.valid && Math.abs(centre.y - 48.0) < 1e-6) { "$centre" }
                val east = map.projectWith(OrbCameraTarget(nan, nan, 1.0, nan, nan, nan, nan, nan, keep), 10.0, 0.0)
                check(Math.abs(east.x - 32.0 - 1024.0 * 10 / 360) < 1e-6) { "$east" }
                map.jumpToFree(map.freeCamera())
                val camera = map.camera()
                check(Math.abs(camera.lon) < 1e-9 && Math.abs(camera.zoom) < 1e-9 && camera.padding.top == 32.0) { "$camera" }
            }
        }
    }
    println("Smoke.kt: ok")
}
