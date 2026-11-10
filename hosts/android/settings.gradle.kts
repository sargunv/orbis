// The Android demo app: orb-android in a NativeActivity, with the web
// demo's data as assets. `cargo xtask android` builds liborbis.so into
// app/src/main/jniLibs; Gradle only packages.

pluginManagement {
  repositories {
    google()
    mavenCentral()
    gradlePluginPortal()
  }
}

dependencyResolutionManagement {
  repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
  repositories {
    google()
    mavenCentral()
  }
}

rootProject.name = "orbis-android"

include(":app")
