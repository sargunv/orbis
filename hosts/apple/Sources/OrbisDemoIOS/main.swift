// The iOS reference app: the web demo's live Liberty over Tokyo Station on
// a MapView, with gestures, a tap popup from queryRenderedFeatures and a
// pin at Tokyo Station that follows the map. `cargo xtask apple --ios-sim`
// builds it into an app for the simulator.

#if canImport(UIKit)
import Orbis
import UIKit

/// A label pinned to a place on the map.
@MainActor
final class Pin {
    let label = UILabel()
    let lon: Double
    let lat: Double

    init(_ text: String, lon: Double, lat: Double, background: UIColor) {
        label.text = text
        label.numberOfLines = 0
        label.font = .systemFont(ofSize: 12)
        label.backgroundColor = background
        label.textColor = .black
        self.lon = lon
        self.lat = lat
    }

    func place(on map: OrbMap) {
        let p = map.project(lon: lon, lat: lat)
        label.isHidden = !p.valid
        let size = label.sizeThatFits(CGSize(width: 260, height: 400))
        label.frame = CGRect(x: p.x - size.width / 2, y: p.y - size.height - 6, width: size.width, height: size.height)
    }
}

final class ViewController: UIViewController {
    var mapView: MapView!
    let status = UILabel()
    var pins: [Pin] = []
    var popup: Pin?
    var pending: (id: UInt64, lon: Double, lat: Double)?
    var started = Date()
    var logged = false

    override func loadView() {
        started = Date()
        let runtime: OrbRuntime
        do {
            runtime = try OrbRuntime.create(jobThreads: 0)
        } catch {
            print("Orbis: no GPU device: \(error)")
            let label = UILabel()
            label.text = "No GPU device: \(error)"
            label.numberOfLines = 0
            view = label
            return
        }
        mapView = MapView(runtime: runtime)
        view = mapView
        let map = mapView.map
        map.jumpTo(target: OrbCameraTarget(lon: 139.767, lat: 35.681, zoom: 13, bearing: .nan, pitch: .nan, roll: .nan, elevation: .nan, fov: .nan, padding: OrbPadding(top: .nan, right: .nan, bottom: .nan, left: .nan)))
        map.loadStyleUrl(url: "https://tiles.openfreemap.org/styles/liberty")
        status.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        status.backgroundColor = UIColor.white.withAlphaComponent(0.8)
        mapView.addSubview(status)
        let station = Pin("Tokyo Station", lon: 139.7671, lat: 35.6812, background: .systemYellow)
        pins.append(station)
        mapView.addSubview(station.label)
        mapView.onFrame = { [unowned self] frame in
            for pin in pins + [popup].compactMap({ $0 }) { pin.place(on: map) }
            status.text = String(format: " z%.2f · %u tiles · %u draws · %@ · frames %d ",
                map.camera().zoom, frame.tilesDrawn, frame.draws, frame.idle ? "idle" : "loading", mapView.framesDrawn)
            status.sizeToFit()
            status.frame.origin = CGPoint(x: 8, y: mapView.bounds.height - status.frame.height - 40)
            if frame.idle && !logged {
                logged = true
                print(String(format: "Orbis: loaded in %.0f ms after %d frames; %u pipelines in %.0f ms",
                    Date().timeIntervalSince(started) * 1000, mapView.framesDrawn,
                    runtime.stats().pipelinesCreated(), runtime.stats().pipelineMs()))
            }
        }
        mapView.onEvent = { [unowned self] event in
            switch event.kind() {
            case .click:
                let at = map.unproject(x: event.x(), y: event.y())
                pending = (map.queryRendered(x0: event.x(), y0: event.y(), x1: event.x(), y1: event.y()), at.lon, at.lat)
            case .queryResult where event.id() == pending?.id:
                showPopup(event.text())
            default:
                break
            }
        }
    }

    func showPopup(_ json: String) {
        popup?.label.removeFromSuperview()
        popup = nil
        guard let query = pending,
              let features = try? JSONSerialization.jsonObject(with: Data(json.utf8)) as? [[String: Any]],
              let feature = features.first
        else { return }
        let layer = (feature["layer"] as? [String: Any])?["id"] as? String ?? "?"
        let properties = (feature["properties"] as? [String: Any] ?? [:])
            .sorted { $0.key < $1.key }.prefix(5).map { "\($0.key): \($0.value)" }
        let pin = Pin(([layer] + properties).joined(separator: "\n"), lon: query.lon, lat: query.lat, background: .white)
        popup = pin
        mapView.addSubview(pin.label)
        pin.place(on: mapView.map)
        print("Orbis: tapped \(layer)")
    }
}

final class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        window = UIWindow(frame: UIScreen.main.bounds)
        window?.rootViewController = ViewController()
        window?.makeKeyAndVisible()
        return true
    }
}

UIApplicationMain(CommandLine.argc, CommandLine.unsafeArgv, nil, NSStringFromClass(AppDelegate.self))
#else
print("OrbisDemoIOS runs on iOS: cargo xtask apple --ios-sim")
#endif
