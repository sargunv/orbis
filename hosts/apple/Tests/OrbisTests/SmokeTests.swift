// The generated Swift's smoke test (plan §14.1): the package links, a map
// draws a style into a texture target, and errors throw with their message.
// The full binding scenario runs in Rust; the emitters' tests cover every
// API shape.

import Foundation
import Orbis
import Testing

private let style = """
    {"version": 8, "center": [0, 0], "zoom": 0,
     "sources": {"box": {"type": "geojson", "data": {
       "type": "Feature", "id": 7, "properties": {},
       "geometry": {"type": "Polygon", "coordinates":
         [[[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]]]}}}},
     "layers": [
       {"id": "background", "type": "background", "paint": {"background-color": "#ff0000"}},
       {"id": "box", "type": "fill", "source": "box", "paint": {"fill-color": "#0000ff"}}]}
    """

@Test func smoke() throws {
    #expect(OrbRuntime.abiVersion() == 3)
    let runtime = try OrbRuntime.create(jobThreads: 0)
    let target = OrbTarget.createTexture(runtime: runtime, width: 64, height: 64, scale: 1)
    let map = OrbMap.create(runtime: runtime, feel: .desktop, clearR: 0, clearG: 0, clearB: 0, clearA: 0)

    let bad = #expect(throws: OrbErrorFailure.self) { try map.setStyleJson(json: "{not json") }
    #expect(bad?.error == .invalidArgument)
    #expect(bad?.message.isEmpty == false)

    // Draws until the blue box shows in the middle.
    try map.setStyleJson(json: style)
    let started = Date()
    let i = (32 * 64 + 32) * 4
    while true {
        _ = try map.render(target: target)
        try target.readPixels()
        if Array(target.pixels()[i..<i + 4]) == [0, 0, 255, 255] { break }
        try #require(Date().timeIntervalSince(started) < 30, "the box never showed")
        Thread.sleep(forTimeInterval: 0.002)
    }
    let hit = try #require(map.hitTest(x: 32, y: 32))
    #expect(hit.layer() == "box" && hit.id() == "7")
}
