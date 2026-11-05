// Video sources' players: AVFoundation plays each video
// the map asks for, and the map view hands each new frame to the map as
// RGBA rows. AVFoundation reads MP4 and QuickTime (H.264, HEVC), not WebM.

import AVFoundation
import CoreVideo
import UniformTypeIdentifiers

@MainActor
final class Videos {
    private var players: [String: Player] = [:]

    /// Whether a video plays, so the display link keeps running for its
    /// frames.
    var playing: Bool { players.values.contains { $0.player.rate != 0 } }

    /// Acts on a video event; false for other events.
    func handle(_ event: OrbEvent, map: OrbMap) -> Bool {
        let source = event.text()
        switch event.kind() {
        case .videoOpen:
            players[source]?.close()
            let urls = event.urls().split(separator: "\n").compactMap { URL(string: String($0)) }
            if let player = Player(urls: urls) {
                players[source] = player
            } else {
                try? map.videoFailed(id: source, message: "AVFoundation plays none of \(urls)")
            }
        case .videoPlay: players[source]?.player.play()
        case .videoPause: players[source]?.player.pause()
        case .videoSeek:
            let time = CMTime(seconds: event.seconds(), preferredTimescale: 600)
            players[source]?.player.seek(to: time, toleranceBefore: .zero, toleranceAfter: .zero)
        case .videoClose: players.removeValue(forKey: source)?.close()
        default: return false
        }
        return true
    }

    /// Hands the map each video's frame if it changed.
    func pump(_ map: OrbMap) {
        for (source, player) in players {
            if player.player.currentItem?.status == .failed {
                players.removeValue(forKey: source)?.close()
                try? map.videoFailed(id: source, message: "AVFoundation can't play this video")
                continue
            }
            guard let (width, height, pixels) = player.newFrame() else { continue }
            try? map.sourceFrame(id: source, width: width, height: height, pixels: pixels)
        }
    }

    func closeAll() {
        players.values.forEach { $0.close() }
        players.removeAll()
    }
}

@MainActor
private final class Player {
    let player: AVPlayer
    let output: AVPlayerItemVideoOutput
    private var looping: NSObjectProtocol?

    /// Plays the first URL AVFoundation reads, muted and looping.
    init?(urls: [URL]) {
        let playable = urls.first { url in
            let types = AVURLAsset.audiovisualTypes().map(\.rawValue)
            return UTType(filenameExtension: url.pathExtension).map { types.contains($0.identifier) } ?? false
        }
        guard let url = playable else { return nil }
        let item = AVPlayerItem(url: url)
        output = AVPlayerItemVideoOutput(pixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        item.add(output)
        player = AVPlayer(playerItem: item)
        player.isMuted = true
        player.actionAtItemEnd = .none
        looping = NotificationCenter.default.addObserver(
            forName: AVPlayerItem.didPlayToEndTimeNotification, object: item, queue: .main
        ) { [player] _ in
            player.seek(to: .zero)
        }
        player.play()
    }

    /// The frame to show now as straight RGBA rows, if it's new.
    func newFrame() -> (UInt32, UInt32, [UInt8])? {
        let time = output.itemTime(forHostTime: CACurrentMediaTime())
        guard output.hasNewPixelBuffer(forItemTime: time),
              let buffer = output.copyPixelBuffer(forItemTime: time, itemTimeForDisplay: nil)
        else { return nil }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let (width, height) = (CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer))
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let bgra = base.assumingMemoryBound(to: UInt8.self)
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let (from, to) = (y * stride + x * 4, (y * width + x) * 4)
                rgba[to] = bgra[from + 2]
                rgba[to + 1] = bgra[from + 1]
                rgba[to + 2] = bgra[from]
                rgba[to + 3] = bgra[from + 3]
            }
        }
        return (UInt32(width), UInt32(height), rgba)
    }

    func close() {
        player.pause()
        player.replaceCurrentItem(with: nil)
        if let looping { NotificationCenter.default.removeObserver(looping) }
    }
}
