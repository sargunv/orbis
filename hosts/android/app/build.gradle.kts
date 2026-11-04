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
            // A reference app: signed with the debug key to install directly.
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    // The web demo's Tokyo extract, style, glyphs and sprites.
    sourceSets["main"].assets.srcDirs("../../../web/demo/data")
    androidResources {
        // PMTiles are read by range: keep them uncompressed in the APK.
        noCompress += "pmtiles"
    }
}

dependencies {
    implementation(project(":orbis"))
}
