// The binding scenario (plan §14.1) through the generated Swift: a style
// into a texture target, a hit test, a query, an edit, a style image,
// camera moves, a click, events, idle and errors, as crates/orb-api/tests/api.rs and the C smoke test do.

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

/// An 8×8 image of one opaque colour.
private func solid(_ r: UInt8, _ g: UInt8, _ b: UInt8) -> [UInt8] {
    Array(repeating: [r, g, b, 255], count: 8 * 8).flatMap { $0 }
}

@Test func scenario() throws {
    #expect(OrbRuntime.abiVersion() == 3)
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

    // A pattern the style lacks is reported; added, it draws, and new
    // pixels replace it in place.
    _ = try map.edit(commands: #"[["setPaintProperty", "box", "fill-pattern", "dot"]]"#)
    #expect(try settle(map, target).contains { $0.kind() == .styleImageMissing && $0.text() == "dot" })
    #expect(!map.hasImage(name: "dot"))
    try map.addImage(name: "dot", width: 8, height: 8, pixels: solid(255, 255, 0), options: #"{"pixelRatio": 1}"#)
    #expect(map.hasImage(name: "dot"))
    let again = #expect(throws: OrbErrorFailure.self) {
        try map.addImage(name: "dot", width: 8, height: 8, pixels: solid(0, 0, 0), options: "")
    }
    #expect(again?.error == .invalidArgument)
    _ = try settle(map, target)
    #expect(try pixel(target, 32, 32) == [255, 255, 0, 255])
    try map.updateImage(name: "dot", width: 8, height: 8, pixels: solid(255, 0, 255))
    _ = try settle(map, target)
    #expect(try pixel(target, 32, 32) == [255, 0, 255, 255])
    try map.removeImage(name: "dot")
    #expect(!map.hasImage(name: "dot"))
    #expect(throws: OrbErrorFailure.self) { try map.removeImage(name: "dot") }
    _ = try map.edit(commands: #"[["setPaintProperty", "box", "fill-pattern", null]]"#)
    _ = try settle(map, target)

    // Camera moves carry a roll and an elevation; unclamped, the centre
    // keeps the elevation it is given.
    map.jumpTo(target: OrbCameraTarget(lon: .nan, lat: .nan, zoom: .nan, bearing: .nan, pitch: .nan, roll: 30, elevation: .nan))
    #expect(map.camera().roll == 30)
    map.setCenterClampedToGround(clamped: false)
    #expect(!map.centerClampedToGround())
    map.setCenterElevation(elevation: 120)
    #expect(map.camera().elevation == 120)
    _ = map.easeTo(target: OrbCameraTarget(lon: .nan, lat: .nan, zoom: .nan, bearing: .nan, pitch: .nan, roll: 0, elevation: 0), durationMs: 50)
    _ = try settle(map, target)
    #expect(map.camera().roll == 0 && map.camera().elevation == 0)
    map.setCenterClampedToGround(clamped: true)

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
