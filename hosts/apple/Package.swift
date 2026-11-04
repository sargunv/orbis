// swift-tools-version: 6.0
// The Apple host: the generated Swift bindings over the C ABI, a map view
// for AppKit and UIKit, and the macOS and iOS reference apps. `cargo xtask apple`
// builds OrbisC.xcframework (the static library and its headers) first.

import PackageDescription

let package = Package(
    name: "Orbis",
    platforms: [.macOS(.v14), .iOS(.v16)],
    products: [
        .library(name: "Orbis", targets: ["Orbis"]),
        .executable(name: "OrbisDemo", targets: ["OrbisDemo"]),
        .executable(name: "OrbisDemoIOS", targets: ["OrbisDemoIOS"]),
    ],
    targets: [
        .binaryTarget(name: "OrbisC", path: "OrbisC.xcframework"),
        .target(
            name: "Orbis",
            dependencies: ["OrbisC"],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("QuartzCore"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("Foundation"),
                .linkedFramework("CoreFoundation"),
                .linkedLibrary("iconv"),
                .linkedLibrary("objc"),
            ]
        ),
        .executableTarget(name: "OrbisDemo", dependencies: ["Orbis"]),
        .executableTarget(name: "OrbisDemoIOS", dependencies: ["Orbis"]),
        .testTarget(name: "OrbisTests", dependencies: ["Orbis"]),
    ]
)
