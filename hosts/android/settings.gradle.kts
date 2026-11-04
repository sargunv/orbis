// The Android host: the `orbis` library (generated Kotlin bindings, the
// JNI shim and liborbis.so from `cargo xtask android`, and a map view) and
// the reference app. Gradle only packages (plan §17).

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
include(":orbis", ":app")
