// The binding scenario (plan §14.1) through the generated Kotlin, on the
// host's JVM: a style into a texture target, a hit test, a query, an edit,
// a click, events, idle and errors, as crates/orb-api/tests/api.rs, the C
// smoke test and the Swift scenario do. `cargo xtask capi --test` runs it.

import dev.orbis.*

const val SIZE = 64u

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

fun pixel(target: OrbTarget, x: Int, y: Int): List<Int> {
    target.readPixels()
    val pixels = target.pixels()
    val i = (y * SIZE.toInt() + x) * 4
    return (i until i + 4).map { pixels[it].toInt() and 0xff }
}

/** Draws until the map needs no frames; the events' kinds. */
fun settle(map: OrbMap, target: OrbTarget): List<OrbEventKind> {
    val started = System.nanoTime()
    while (map.needsFrame()) {
        map.render(target)
        check(System.nanoTime() - started < 30_000_000_000) { "never idle" }
        Thread.sleep(2)
    }
    val kinds = mutableListOf<OrbEventKind>()
    while (true) {
        val event = map.pollEvent() ?: break
        event.use { kinds += it.kind() }
    }
    return kinds
}

fun main() {
    check(OrbRuntime.abiVersion() == 1u)
    OrbRuntime.create(0u).use { runtime ->
        OrbTarget.createTexture(runtime, SIZE, SIZE, 1.0).use { target ->
            OrbMap.create(runtime, OrbFeel.Desktop, 0.0, 0.0, 0.0, 0.0).use { map ->
                val bad = runCatching { map.setStyleJson("{not json") }.exceptionOrNull()
                check(bad is OrbErrorException && bad.error == OrbError.InvalidArgument && !bad.message.isNullOrEmpty())
                check(runCatching { map.edit("[]") }.exceptionOrNull() is OrbErrorException)

                map.setStyleJson(STYLE)
                val kinds = settle(map, target)
                check(kinds.first() == OrbEventKind.StyleLoad && OrbEventKind.Load in kinds && kinds.last() == OrbEventKind.Idle) { "$kinds" }
                check(kotlin.math.abs(map.camera().zoom) < 1e-9)
                check(pixel(target, 32, 32) == listOf(0, 0, 255, 255))
                check(pixel(target, 1, 1) == listOf(255, 0, 0, 255))

                map.hitTest(32.0, 32.0)!!.use { hit ->
                    check(hit.layer() == "box" && hit.hasId() && hit.id() == "7")
                }
                check(map.hitTest(1.0, 1.0) == null)

                val query = map.queryRendered(32.0, 32.0, 32.0, 32.0)
                var features: String? = null
                repeat(5000) {
                    if (features != null) return@repeat
                    while (true) {
                        val event = map.pollEvent() ?: break
                        event.use {
                            if (it.kind() == OrbEventKind.QueryResult && it.id() == query) {
                                check(it.flag())
                                features = it.text()
                            }
                        }
                    }
                    Thread.sleep(1)
                }
                check(features?.contains("\"id\":\"box\"") == true) { "$features" }
                settle(map, target)

                val generation = map.edit("""[["setPaintProperty", "box", "fill-color", "#00ff00"]]""")
                var shown = 0uL
                while (map.needsFrame()) shown = maxOf(shown, map.render(target).generation)
                check(shown >= generation)
                check(pixel(target, 32, 32) == listOf(0, 255, 0, 255))

                val none = OrbModifiers(false, false, false, false)
                map.pointerDown(1u, OrbPointerKind.Mouse, OrbButton.Primary, 10.0, 20.0, none)
                map.pointerUp(1u, 10.0, 20.0)
                check(OrbEventKind.Click in settle(map, target))
                check(!map.needsFrame())

                val p = map.project(5.0, 5.0)
                val back = map.unproject(p.x, p.y)
                check(p.valid && back.valid && kotlin.math.abs(back.lon - 5) < 1e-6 && kotlin.math.abs(back.lat - 5) < 1e-6)
            }
        }
    }
    println("Scenario.kt: ok")
}
