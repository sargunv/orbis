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
    }
  }
  // The web demo's Tokyo extract, styles, glyphs and sprites, and the
  // index of them that `cargo xtask android` writes.
  sourceSets["main"].assets.srcDirs("../../../web/demo/data", "build/generated/assets")
  androidResources {
    // PMTiles are read by range: keep them uncompressed in the APK.
    noCompress += "pmtiles"
  }
}
