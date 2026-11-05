// The macOS reference app: the web demo's style and behaviour on a
// MapView. Gestures move the map; a click shows the features under it in a
// popup anchored to the map; hovering names the feature under the pointer;
// a pin marks Tokyo Station. `OrbisDemo [tokyo | style URL or path]`: the
// web demo's live OpenFreeMap Liberty over Tokyo by default, `tokyo` for
// the committed extract.
//
// `OrbisDemo --snapshot out.png [style]` draws the map into a texture
// target instead (plan §5.4's owned texture), until it is idle, with the
// pin drawn on, writes a PNG and prints timings and pipeline counts.

import AppKit
import Orbis

/// The web demo's data directory in this checkout.
let demoData = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .appendingPathComponent("../../../../web/demo/data")
    .standardizedFileURL

/// The web demo's default: live Liberty over Tokyo Station.
let liveStyle = "https://tiles.openfreemap.org/styles/liberty"

/// Loads the style an argument names: nothing for live Liberty at the web
/// demo's camera, `tokyo` for the extract, or a URL or path.
func loadStyle(_ argument: String?, into map: OrbMap) throws {
    switch argument {
    case nil:
        map.jumpTo(target: OrbCameraTarget(lon: 139.767, lat: 35.681, zoom: 13, bearing: .nan, pitch: .nan, roll: .nan, elevation: .nan))
        map.loadStyleUrl(url: liveStyle)
    case "tokyo":
        try map.setStyleJson(json: try tokyoStyle())
    case let path?:
        map.loadStyleUrl(url: path.contains("://") ? path : "file://\(path)")
    }
}

/// The Tokyo style with its tiles, glyphs and sprites as file URLs.
func tokyoStyle() throws -> String {
    let json = try String(contentsOf: demoData.appendingPathComponent("tokyo-mvt.json"), encoding: .utf8)
    return json
        .replacingOccurrences(of: "pmtiles://./data/", with: "pmtiles://file://\(demoData.path)/")
        .replacingOccurrences(of: "\"./data/", with: "\"file://\(demoData.path)/")
}

/// A label pinned to a place on the map.
@MainActor
final class Annotation {
    let view: NSTextField
    let lon: Double
    let lat: Double

    init(text: String, lon: Double, lat: Double, background: NSColor) {
        view = NSTextField(labelWithString: text)
        view.font = .systemFont(ofSize: 12)
        view.drawsBackground = true
        view.backgroundColor = background
        view.textColor = .black
        view.wantsLayer = true
        view.layer?.cornerRadius = 4
        view.sizeToFit()
        self.lon = lon
        self.lat = lat
    }

    /// Puts the label's bottom centre on its place.
    func place(on map: OrbMap) {
        let p = map.project(lon: lon, lat: lat)
        view.isHidden = !p.valid
        let size = view.fittingSize
        view.frame = NSRect(x: p.x - size.width / 2, y: p.y - size.height - 6, width: size.width, height: size.height)
    }
}

/// The map view with the demo's hover, click and annotations.
final class DemoView: MapView {
    let status = NSTextField(labelWithString: "")
    var annotations: [Annotation] = []
    var popup: Annotation?
    var pendingQuery: (id: UInt64, lon: Double, lat: Double)?
    var tracking: NSTrackingArea?

    override init(runtime: OrbRuntime, frame: CGRect) {
        super.init(runtime: runtime, frame: frame)
        status.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        status.drawsBackground = true
        status.backgroundColor = NSColor.white.withAlphaComponent(0.8)
        addSubview(status)
        let station = Annotation(text: "Tokyo Station", lon: 139.7671, lat: 35.6812, background: .systemYellow)
        annotations.append(station)
        addSubview(station.view)
        onFrame = { [unowned self] frame in
            for a in annotations + [popup].compactMap({ $0 }) {
                a.place(on: map)
            }
            let c = map.camera()
            status.stringValue = String(
                format: " z%.2f · %d tiles · %d draws · %@ · frames %d ",
                c.zoom, frame.tilesDrawn, frame.draws, frame.idle ? "idle" : "loading", framesDrawn)
            status.sizeToFit()
            status.frame.origin = CGPoint(x: 8, y: 8)
        }
        onEvent = { [unowned self] event in
            switch event.kind() {
            case .click:
                let lngLat = map.unproject(x: event.x(), y: event.y())
                let id = map.queryRendered(x0: event.x(), y0: event.y(), x1: event.x(), y1: event.y())
                pendingQuery = (id, lngLat.lon, lngLat.lat)
            case .queryResult where event.id() == pendingQuery?.id:
                showPopup(event.text())
            case .error:
                print("orbis: \(event.text())")
            default:
                break
            }
        }
    }

    /// A popup with the topmost feature's layer and properties.
    func showPopup(_ json: String) {
        popup?.view.removeFromSuperview()
        popup = nil
        guard let query = pendingQuery,
              let features = try? JSONSerialization.jsonObject(with: Data(json.utf8)) as? [[String: Any]],
              let feature = features.first
        else { return }
        let layer = (feature["layer"] as? [String: Any])?["id"] as? String ?? "?"
        let properties = (feature["properties"] as? [String: Any] ?? [:])
            .sorted { $0.key < $1.key }
            .prefix(6)
            .map { "\($0.key): \($0.value)" }
        let annotation = Annotation(
            text: ([layer] + properties).joined(separator: "\n"),
            lon: query.lon, lat: query.lat, background: .white)
        popup = annotation
        addSubview(annotation.view)
        annotation.place(on: map)
        print("clicked \(layer): \(properties.joined(separator: ", "))")
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let tracking { removeTrackingArea(tracking) }
        let area = NSTrackingArea(rect: bounds, options: [.mouseMoved, .activeInKeyWindow, .inVisibleRect], owner: self)
        addTrackingArea(area)
        tracking = area
    }

    override func mouseMoved(with event: NSEvent) {
        let p = convert(event.locationInWindow, from: nil)
        if let hit = map.hitTest(x: p.x, y: p.y) {
            window?.subtitle = "\(hit.layer())\(hit.hasId() ? " #\(hit.id())" : "")"
        } else {
            window?.subtitle = ""
        }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let runtime: OrbRuntime
        do {
            runtime = try OrbRuntime.createWith(
                jobThreads: 0, backend: .auto, stateDir: stateDir(), androidApiLevel: 0)
        } catch {
            fatalError("no GPU device: \(error)")
        }
        let view = DemoView(runtime: runtime, frame: NSRect(x: 0, y: 0, width: 1024, height: 768))
        do {
            try loadStyle(CommandLine.arguments.dropFirst().first, into: view.map)
        } catch {
            fatalError("style: \(error)")
        }
        window = NSWindow(
            contentRect: view.frame, styleMask: [.titled, .closable, .resizable, .miniaturizable],
            backing: .buffered, defer: false)
        window.title = "Orbis"
        window.contentView = view
        window.makeFirstResponder(view)
        window.center()
        window.makeKeyAndOrderFront(nil)
        // Launched from a terminal, the app may not be activated; show the
        // window anyway (wgpu draws nothing into an occluded window).
        window.orderFrontRegardless()
        NSApp.activate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

/// Where the runtime keeps state between runs: the crash sentinel and the
/// HTTP cache.
func stateDir() -> String {
    FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("dev.orbis.demo").path
}

/// Renders into a texture until idle and writes a PNG with the pin.
@MainActor
func snapshot(to path: String, style: String?) throws {
    let started = Date()
    let runtime = try OrbRuntime.createWith(
        jobThreads: 0, backend: .auto, stateDir: stateDir(), androidApiLevel: 0)
    let deviceMs = Date().timeIntervalSince(started) * 1000
    let (width, height, scale) = (1024, 768, 2.0)
    let target = OrbTarget.createTexture(
        runtime: runtime, width: UInt32(Double(width) * scale), height: UInt32(Double(height) * scale),
        scale: scale)
    let map = OrbMap.create(runtime: runtime, feel: .desktop, clearR: 1, clearG: 1, clearB: 1, clearA: 1)
    try loadStyle(style, into: map)
    var frames = 0
    var slowest = 0.0
    var renderMs = 0.0
    var last = OrbFrame(drawn: false, generation: 0, idle: false, loaded: false, tilesDrawn: 0, tilesLoading: 0, draws: 0, dispatches: 0)
    while map.needsFrame() {
        let frameStarted = Date()
        last = try map.render(target: target)
        let ms = Date().timeIntervalSince(frameStarted) * 1000
        renderMs += ms
        slowest = max(slowest, ms)
        frames += 1
        while let event = map.pollEvent() {
            if event.kind() == .error { print("orbis: \(event.text())") }
        }
        if Date().timeIntervalSince(started) > 60 { throw CocoaError(.featureUnsupported) }
        Thread.sleep(forTimeInterval: 0.002)
    }
    let totalMs = Date().timeIntervalSince(started) * 1000
    try target.readPixels()
    let pixels = target.pixels()
    let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: Int(target.width()), pixelsHigh: Int(target.height()),
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: Int(target.width()) * 4, bitsPerPixel: 32)!
    pixels.withUnsafeBufferPointer { rep.bitmapData!.update(from: $0.baseAddress!, count: pixels.count) }
    // The pin, as the window draws it: a label above the projected point.
    let context = NSGraphicsContext(bitmapImageRep: rep)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = context
    let flip = NSAffineTransform()
    flip.translateX(by: 0, yBy: CGFloat(target.height()))
    flip.scaleX(by: scale, yBy: -scale)
    flip.concat()
    let p = map.project(lon: 139.7671, lat: 35.6812)
    if p.valid {
        let text = NSAttributedString(string: " Tokyo Station ", attributes: [
            .font: NSFont.systemFont(ofSize: 12), .backgroundColor: NSColor.systemYellow,
        ])
        let size = text.size()
        let flipped = NSAffineTransform()
        flipped.translateX(by: p.x - size.width / 2, yBy: p.y - 6)
        flipped.scaleX(by: 1, yBy: -1)
        flipped.concat()
        text.draw(at: .zero)
    }
    NSGraphicsContext.restoreGraphicsState()
    try rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: path))
    print(String(
        format: "snapshot: %@ after %d frames, %.0f ms (device %.0f ms, rendering %.0f ms, slowest frame %.0f ms); %u pipelines in %.0f ms; %u tiles, %u draws, generation %llu",
        path, frames, totalMs, deviceMs, renderMs, slowest, runtime.pipelinesCreated(), runtime.pipelineMs(),
        last.tilesDrawn, last.draws, last.generation))
}

let arguments = Array(CommandLine.arguments.dropFirst())
if arguments.first == "--snapshot", arguments.count >= 2 {
    do {
        try MainActor.assumeIsolated {
            try snapshot(to: arguments[1], style: arguments.count > 2 ? arguments[2] : nil)
        }
        exit(0)
    } catch {
        print("snapshot failed: \(error)")
        exit(1)
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
