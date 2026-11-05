// The generated Swift's smoke test (plan §14.1): the package links, a map
// draws a style into a texture target and into a CAMetalLayer, and errors
// throw with their message.
// The full binding scenario runs in Rust; the emitters' tests cover every
// API shape.

import Foundation
import Orbis
import QuartzCore
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
    #expect(OrbRuntime.abiVersion() == 8)
    let runtime = try OrbRuntime.create(jobThreads: 0)
    let target = OrbTarget.createTexture(runtime: runtime, width: 64, height: 64, scale: 1)
    let map = OrbMap.create(runtime: runtime, feel: .desktop, clearR: 0, clearG: 0, clearB: 0, clearA: 0)

    let bad = #expect(throws: OrbErrorFailure.self) { try map.setStyleJson(json: "{not json") }
    #expect(bad?.error == .invalidArgument)
    #expect(bad?.message.isEmpty == false)

    // Draws until the blue box shows in the middle.
    // From a style a request rule finds on disk (files read offline).
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("orb-smoke-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    try style.write(to: dir.appendingPathComponent("style.json"), atomically: true, encoding: .utf8)
    #expect(throws: OrbErrorFailure.self) { try runtime.setRequestRules(rules: "{bad") }
    try runtime.setRequestRules(rules: "{\"rewrites\": [[\"smoke://\", \"file://\(dir.path)/\"]]}")
    runtime.setOnline(online: false)
    #expect(!runtime.online())
    map.loadStyleUrl(url: "smoke://style.json")
    let started = Date()
    let i = (32 * 64 + 32) * 4
    while true {
        _ = try map.render(target: target, frameTimeMs: 0)
        try target.readPixels()
        if Array(target.pixels()[i..<i + 4]) == [0, 0, 255, 255] { break }
        try #require(Date().timeIntervalSince(started) < 30, "the box never showed")
        Thread.sleep(forTimeInterval: 0.002)
    }
    let hit = try #require(map.hitTest(x: 32, y: 32))
    #expect(hit.layer() == "box" && hit.id() == "7")

    // Edits built one by one apply as one generation; a value that isn't
    // JSON throws at once.
    let tx = OrbTransaction.create()
    try tx.setPaintProperty(layer: "box", name: "fill-color", value: "\"#00ff00\"")
    try tx.setPaintProperty(layer: "box", name: "fill-opacity", value: "1")
    #expect(throws: OrbErrorFailure.self) { try tx.setFilter(layer: "box", filter: "[==") }
    let before = map.generation()
    #expect(try map.edit(transaction: tx) == before + 1)

    // Padding moves the centre down; a camera's projection answers before
    // any move; the eye round-trips.
    let keep = OrbPadding(top: .nan, right: .nan, bottom: .nan, left: .nan)
    var padding = keep
    padding.top = 32
    map.jumpTo(target: OrbCameraTarget(lon: .nan, lat: .nan, zoom: .nan, bearing: .nan, pitch: .nan, roll: .nan, elevation: .nan, fov: .nan, padding: padding))
    let centre = map.project(lon: 0, lat: 0)
    #expect(centre.valid && abs(centre.y - 48) < 1e-6)
    let closer = OrbCameraTarget(lon: .nan, lat: .nan, zoom: 1, bearing: .nan, pitch: .nan, roll: .nan, elevation: .nan, fov: .nan, padding: keep)
    let east = map.projectWith(camera: closer, lon: 10, lat: 0)
    #expect(abs(east.x - 32 - 1024 * 10 / 360) < 1e-6)
    map.jumpToFree(camera: map.freeCamera())
    let camera = map.camera()
    #expect(abs(camera.lon) < 1e-9 && abs(camera.zoom) < 1e-9 && camera.padding.top == 32)
}

/// A window-style target: a CAMetalLayer, presented frame by frame, as
/// `MapView` draws. Frames acquire and present its drawables on the same
/// device borrow as the map's render, until the map is idle.
/// A layer's frames on a thread of the app's own (Swift Testing runs this off
/// the main thread): a runtime and what's made with it belong to the thread
/// that made them. A layer inside a window's view must still present on the
/// main thread on macOS: wgpu-hal asks the view for its window on acquire
/// (#35).
@Test func metalLayerFrames() throws {
    #expect(!Thread.isMainThread)
    let runtime = try OrbRuntime.create(jobThreads: 0)
    if runtime.stats().compat() { return } // Compat draws into EGL windows.
    let layer = CAMetalLayer()
    layer.drawableSize = CGSize(width: 64, height: 64)
    let address = UInt(bitPattern: Unmanaged.passUnretained(layer).toOpaque())
    let target = try OrbTarget.createMetalLayer(runtime: runtime, layer: address, width: 64, height: 64, scale: 1)
    let map = OrbMap.create(runtime: runtime, feel: .desktop, clearR: 0, clearG: 0, clearB: 0, clearA: 0)
    try map.setStyleJson(json: style)
    let started = Date()
    var drawn = 0
    repeat {
        if try map.render(target: target, frameTimeMs: ProcessInfo.processInfo.systemUptime * 1000).drawn {
            drawn += 1
        }
        while map.pollEvent() != nil {}
        try #require(Date().timeIntervalSince(started) < 30, "the map never idled")
        Thread.sleep(forTimeInterval: 0.002)
    } while map.needsFrame() || drawn < 3
    #expect(drawn >= 3)
}

