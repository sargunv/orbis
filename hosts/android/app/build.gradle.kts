plugins {
  id("com.android.application")
}

android {
  namespace = "dev.orbis.demo"
  compileSdk = 36
  defaultConfig {
    applicationId = "dev.orbis.demo"
    minSdk = 24
    targetSdk = 36
    versionCode = 1
    versionName = "0.0.0"
  }
  buildTypes {
    release {
      // A demo app: signed with the debug key to install directly.
      signingConfig = signingConfigs.getByName("debug")
      // R8 keeps of the Kotlin stdlib only what the verifier calls.
      isMinifyEnabled = true
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
    }
  }
  // The web demo's Tokyo extract, styles, glyphs and sprites, which the
  // app reads in place as file:///android_asset/ URLs.
  sourceSets["main"].assets.srcDirs("../../../web/demo/data")
  androidResources {
    // PMTiles are read by range: keep them uncompressed in the APK.
    noCompress += "pmtiles"
  }
}

dependencies {
  // rustls-platform-verifier's Kotlin half, through which HTTPS verifies
  // with Android's trust store. Its version must be the one of
  // rustls-platform-verifier-android in Cargo.lock, which `cargo xtask
  // android` passes; it declares no dependency on the stdlib it calls.
  implementation(
    "org.rustls:rustls-platform-verifier:${providers.gradleProperty("rustlsPlatformVerifier").get()}",
  )
  implementation("org.jetbrains.kotlin:kotlin-stdlib:2.3.21")
}
