// The AVFoundation video player end to end: a map asks
// for a video source, `Videos` plays an MP4 written here, and its frames
// draw.

import AVFoundation
import Foundation
@testable import Orbis
import Testing

/// Writes `seconds` of one opaque colour as H.264 in an MP4.
private func solidVideo(at url: URL, rgb: (UInt8, UInt8, UInt8), seconds: Int) async throws {
    let (width, height) = (64, 64)
    let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
    ])
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
    ])
    writer.add(input)
    writer.startWriting()
    writer.startSession(atSourceTime: .zero)
    for frame in 0..<(seconds * 10) {
        while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 1_000_000) }
        var buffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &buffer)
        let pixels = try #require(buffer)
        CVPixelBufferLockBaseAddress(pixels, [])
        let base = CVPixelBufferGetBaseAddress(pixels)!.assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(pixels)
        for y in 0..<height {
            for x in 0..<width {
                let at = y * stride + x * 4
                (base[at], base[at + 1], base[at + 2], base[at + 3]) = (rgb.2, rgb.1, rgb.0, 255)
            }
        }
        CVPixelBufferUnlockBaseAddress(pixels, [])
        adaptor.append(pixels, withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: 10))
    }
    input.markAsFinished()
    await writer.finishWriting()
    #expect(writer.status == .completed)
}

@MainActor
@Test func avFoundationPlaysVideoSources() async throws {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let mp4 = dir.appendingPathComponent("green.mp4")
    try await solidVideo(at: mp4, rgb: (0, 200, 0), seconds: 2)

    let runtime = try OrbRuntime.create(jobThreads: 0)
    let target = OrbTarget.createTexture(runtime: runtime, width: 64, height: 64, scale: 1)
    let map = OrbMap.create(runtime: runtime, clearR: 0, clearG: 0, clearB: 0, clearA: 0)
    map.setPlaysVideo(plays: true)
    // WebM comes first: AVFoundation skips it for the MP4.
    try map.setStyleJson(json: """
        {"version": 8, "center": [0, 0], "zoom": 0,
         "sources": {"v": {"type": "video",
           "urls": ["\(dir.appendingPathComponent("green.webm").absoluteString)", "\(mp4.absoluteString)"],
           "coordinates": [[-170, 80], [170, 80], [170, -80], [-170, -80]]}},
         "layers": [{"id": "v", "type": "raster", "source": "v", "paint": {"raster-fade-duration": 0}}]}
        """)
    let videos = Videos()
    defer { videos.closeAll() }
    var centre: [UInt8] = []
    let started = Date()
    while Date().timeIntervalSince(started) < 20 {
        videos.pump(map)
        if map.needsFrame() {
            _ = try map.render(target: target, frameTimeMs: ProcessInfo.processInfo.systemUptime * 1000)
        }
        while let event = map.pollEvent() {
            _ = videos.handle(event, map: map)
            #expect(event.kind() != .error, "\(event.text())")
        }
        try target.readPixels()
        centre = Array(target.pixels()[(32 * 64 + 32) * 4..<(32 * 64 + 32) * 4 + 4])
        if centre[1] > 150 { break }
        try await Task.sleep(nanoseconds: 10_000_000)
    }
    // H.264 shifts colours a little.
    #expect(centre[0] < 30 && centre[1] > 150 && centre[2] < 30 && centre[3] == 255, "\(centre)")
}
