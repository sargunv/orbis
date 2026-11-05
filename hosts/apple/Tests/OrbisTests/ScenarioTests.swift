// The binding scenario (plan §14.1) through the generated Swift: a style
// into a texture target, a hit test, a query, an edit, a click, events,
// idle and errors, as crates/orb-api/tests/api.rs and the C smoke test do.

import Foundation
import Orbis
import Testing

private let size: UInt32 = 64

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

private func pixel(_ target: OrbTarget, _ x: Int, _ y: Int) throws -> [UInt8] {
    try target.readPixels()
    let pixels = target.pixels()
    let i = (y * Int(size) + x) * 4
    return Array(pixels[i..<i + 4])
}

/// Draws until the map needs no frames; returns the events.
private func settle(_ map: OrbMap, _ target: OrbTarget) throws -> [OrbEvent] {
    let started = Date()
    while map.needsFrame() {
        _ = try map.render(target: target)
        #expect(Date().timeIntervalSince(started) < 30)
        Thread.sleep(forTimeInterval: 0.002)
    }
    var events: [OrbEvent] = []
    while let event = map.pollEvent() {
        events.append(event)
    }
    return events
}

@Test func scenario() throws {
    #expect(OrbRuntime.abiVersion() == 2)
    let runtime = try OrbRuntime.create(jobThreads: 0)
    let target = OrbTarget.createTexture(runtime: runtime, width: size, height: size, scale: 1)
    let map = OrbMap.create(runtime: runtime, feel: .desktop, clearR: 0, clearG: 0, clearB: 0, clearA: 0)

    let bad = #expect(throws: OrbErrorFailure.self) { try map.setStyleJson(json: "{not json") }
    #expect(bad?.error == .invalidArgument)
    #expect(bad?.message.isEmpty == false)
    #expect(throws: OrbErrorFailure.self) { _ = try map.edit(commands: "[]") }

    try map.setStyleJson(json: style)
    let kinds = try settle(map, target).map { $0.kind() }
    #expect(kinds.first == .styleLoad)
    #expect(kinds.contains(.load))
    #expect(kinds.last == .idle)
    #expect(abs(map.camera().zoom) < 1e-9)
    #expect(try pixel(target, 32, 32) == [0, 0, 255, 255])
    #expect(try pixel(target, 1, 1) == [255, 0, 0, 255])

    let hit = try #require(map.hitTest(x: 32, y: 32))
    #expect(hit.layer() == "box")
    #expect(hit.hasId() && hit.id() == "7")
    #expect(map.hitTest(x: 1, y: 1) == nil)

    let query = map.queryRendered(x0: 32, y0: 32, x1: 32, y1: 32)
    var features: String?
    for _ in 0..<5000 where features == nil {
        while let event = map.pollEvent() {
            if event.kind() == .queryResult && event.id() == query {
                #expect(event.flag())
                features = event.text()
            }
        }
        Thread.sleep(forTimeInterval: 0.001)
    }
    #expect(features?.contains(#""id":"box""#) == true)
    _ = try settle(map, target)

    let generation = try map.edit(commands: ##"[["setPaintProperty", "box", "fill-color", "#00ff00"]]"##)
    var shown: UInt64 = 0
    while map.needsFrame() {
        shown = max(shown, try map.render(target: target).generation)
    }
    #expect(shown >= generation)
    #expect(try pixel(target, 32, 32) == [0, 255, 0, 255])

    let none = OrbModifiers(shift: false, ctrl: false, alt: false, meta: false)
    _ = map.pointerDown(id: 1, kind: .mouse, button: .primary, x: 10, y: 20, modifiers: none)
    _ = map.pointerUp(id: 1, x: 10, y: 20)
    let clicks = try settle(map, target).filter { $0.kind() == .click }
    #expect(clicks.count == 1 && clicks.first?.x() == 10 && clicks.first?.y() == 20)
    #expect(!map.needsFrame())

    let p = map.project(lon: 5, lat: 5)
    let back = map.unproject(x: p.x, y: p.y)
    #expect(p.valid && back.valid && abs(back.lon - 5) < 1e-6 && abs(back.lat - 5) < 1e-6)
}
