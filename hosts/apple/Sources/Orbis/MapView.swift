// A view that draws an Orbis map into a CAMetalLayer, paced by the
// display, and hands pointer, scroll, key and touch input to the map's
// gestures. It draws only while the map needs frames: an idle map costs no
// frames, and data arriving on Orbis's threads wakes the display link.

import QuartzCore

#if canImport(AppKit)
import AppKit
public typealias PlatformView = NSView
#elseif canImport(UIKit)
import UIKit
public typealias PlatformView = UIView
#endif

@MainActor
open class MapView: PlatformView {
    public let runtime: OrbRuntime
    public let map: OrbMap
    /// Called after each frame drawn, with what it did.
    public var onFrame: ((OrbFrame) -> Void)?
    /// Called with each event the map, its camera or a gesture reports.
    public var onEvent: ((OrbEvent) -> Void)?
    /// Frames drawn since the view was made.
    public private(set) var framesDrawn = 0

    private let metalLayer = CAMetalLayer()
    private var target: OrbTarget?
    private var displayLink: CADisplayLink?

    public init(runtime: OrbRuntime, frame: CGRect = .zero) {
        self.runtime = runtime
        #if os(macOS)
        let feel = OrbFeel.desktop
        #else
        let feel = OrbFeel.ios
        #endif
        map = OrbMap.create(runtime: runtime, feel: feel, clearR: 1, clearG: 1, clearB: 1, clearA: 1)
        super.init(frame: frame)
        #if os(macOS)
        wantsLayer = true
        #else
        metalLayer.frame = bounds
        layer.addSublayer(metalLayer)
        isMultipleTouchEnabled = true
        #endif
        map.setWake { [weak self] in
            Task { @MainActor in self?.setNeedsFrame() }
        }
    }

    @available(*, unavailable)
    public required init?(coder: NSCoder) {
        fatalError("MapView is made in code")
    }

    /// Draws frames until the map needs none; call after changing the map.
    public func setNeedsFrame() {
        displayLink?.isPaused = false
    }

    // MARK: Layer, size and display link

    #if os(macOS)
    override open func makeBackingLayer() -> CALayer { metalLayer }
    override open var isFlipped: Bool { true }
    override open var acceptsFirstResponder: Bool { true }

    override open func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        attach()
    }

    override open func viewDidChangeBackingProperties() {
        super.viewDidChangeBackingProperties()
        resizeTarget()
    }

    override open func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        resizeTarget()
    }

    private var scale: CGFloat { window?.backingScaleFactor ?? 2 }
    #else
    override open func didMoveToWindow() {
        super.didMoveToWindow()
        attach()
    }

    override open func layoutSubviews() {
        super.layoutSubviews()
        metalLayer.frame = bounds
        resizeTarget()
    }

    private var scale: CGFloat { window?.screen.scale ?? UIScreen.main.scale }
    #endif

    private func attach() {
        guard window != nil else {
            displayLink?.invalidate()
            displayLink = nil
            return
        }
        guard displayLink == nil else { return }
        #if os(macOS)
        let link = displayLink(target: self, selector: #selector(step))
        #else
        let link = CADisplayLink(target: self, selector: #selector(step))
        #endif
        link.add(to: .main, forMode: .common)
        displayLink = link
        resizeTarget()
    }

    private func resizeTarget() {
        guard window != nil else { return }
        let scale = self.scale
        metalLayer.contentsScale = scale
        let width = UInt32(max(1, (bounds.width * scale).rounded()))
        let height = UInt32(max(1, (bounds.height * scale).rounded()))
        if let target {
            target.resize(width: width, height: height, scale: Double(scale))
        } else {
            let address = UInt(bitPattern: Unmanaged.passUnretained(metalLayer).toOpaque())
            do {
                target = try OrbTarget.createMetalLayer(
                    runtime: runtime, layer: address, width: width, height: height, scale: Double(scale))
            } catch {
                print("Orbis: no surface: \(error)")
            }
        }
        setNeedsFrame()
    }

    @objc private func step(_ link: CADisplayLink) {
        guard let target else { return }
        do {
            let frame = try map.render(target: target)
            if frame.drawn {
                framesDrawn += 1
                onFrame?(frame)
            }
        } catch let failure as OrbErrorFailure where failure.error == .targetLost {
            self.target = nil
            resizeTarget()
        } catch {
            print("Orbis: \(error)")
        }
        while let event = map.pollEvent() {
            onEvent?(event)
        }
        if !map.needsFrame() {
            link.isPaused = true
        }
    }

    // MARK: Input

    private func used(_ used: Bool) {
        if used { setNeedsFrame() }
    }

    #if os(macOS)
    private func point(_ event: NSEvent) -> CGPoint {
        convert(event.locationInWindow, from: nil)
    }

    private func modifiers(_ event: NSEvent) -> OrbModifiers {
        let flags = event.modifierFlags
        return OrbModifiers(
            shift: flags.contains(.shift), ctrl: flags.contains(.control),
            alt: flags.contains(.option), meta: flags.contains(.command))
    }

    private func down(_ event: NSEvent, _ button: OrbButton) {
        let p = point(event)
        used(map.pointerDown(
            id: 0, kind: .mouse, button: button, x: p.x, y: p.y, modifiers: modifiers(event)))
    }

    private func moved(_ event: NSEvent) {
        let p = point(event)
        used(map.pointerMove(id: 0, x: p.x, y: p.y))
    }

    private func up(_ event: NSEvent) {
        let p = point(event)
        used(map.pointerUp(id: 0, x: p.x, y: p.y))
    }

    override open func mouseDown(with event: NSEvent) { down(event, .primary) }
    override open func mouseDragged(with event: NSEvent) { moved(event) }
    override open func mouseUp(with event: NSEvent) { up(event) }
    override open func rightMouseDown(with event: NSEvent) { down(event, .secondary) }
    override open func rightMouseDragged(with event: NSEvent) { moved(event) }
    override open func rightMouseUp(with event: NSEvent) { up(event) }
    override open func otherMouseDown(with event: NSEvent) { down(event, .middle) }
    override open func otherMouseDragged(with event: NSEvent) { moved(event) }
    override open func otherMouseUp(with event: NSEvent) { up(event) }

    override open func scrollWheel(with event: NSEvent) {
        let p = point(event)
        // DOM wheel deltas: positive y scrolls down, which zooms out.
        let unit: OrbWheelUnit = event.hasPreciseScrollingDeltas ? .pixel : .line
        used(map.wheel(
            x: p.x, y: p.y, dx: -event.scrollingDeltaX, dy: -event.scrollingDeltaY,
            unit: unit, modifiers: modifiers(event)))
    }

    override open func magnify(with event: NSEvent) {
        // As browsers report a trackpad pinch: a Ctrl-wheel.
        let p = point(event)
        var mods = modifiers(event)
        mods.ctrl = true
        used(map.wheel(
            x: p.x, y: p.y, dx: 0, dy: -event.magnification * 100, unit: .pixel, modifiers: mods))
    }

    override open func keyDown(with event: NSEvent) {
        let key: OrbKey? = switch event.keyCode {
        case 123: .left
        case 124: .right
        case 125: .down
        case 126: .up
        default:
            switch event.charactersIgnoringModifiers {
            case "+", "=": .zoomIn
            case "-": .zoomOut
            default: nil
            }
        }
        if let key {
            used(map.key(key: key, modifiers: modifiers(event)))
        } else {
            super.keyDown(with: event)
        }
    }

    override open func resignFirstResponder() -> Bool {
        used(map.blur())
        return super.resignFirstResponder()
    }
    #else
    private var touchIds: [ObjectIdentifier: UInt32] = [:]
    private var nextTouchId: UInt32 = 0

    private func id(_ touch: UITouch) -> UInt32 {
        let key = ObjectIdentifier(touch)
        if let id = touchIds[key] { return id }
        nextTouchId &+= 1
        touchIds[key] = nextTouchId
        return nextTouchId
    }

    private static let noModifiers = OrbModifiers(shift: false, ctrl: false, alt: false, meta: false)

    override open func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        for touch in touches {
            let p = touch.location(in: self)
            used(map.pointerDown(
                id: id(touch), kind: .touch, button: .primary, x: p.x, y: p.y,
                modifiers: Self.noModifiers))
        }
    }

    override open func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        for touch in touches {
            let p = touch.location(in: self)
            used(map.pointerMove(id: id(touch), x: p.x, y: p.y))
        }
    }

    override open func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        for touch in touches {
            let p = touch.location(in: self)
            used(map.pointerUp(id: id(touch), x: p.x, y: p.y))
            touchIds[ObjectIdentifier(touch)] = nil
        }
    }

    override open func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        for touch in touches {
            used(map.pointerCancel(id: id(touch)))
            touchIds[ObjectIdentifier(touch)] = nil
        }
    }
    #endif
}
