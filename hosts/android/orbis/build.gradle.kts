plugins {
  id("com.android.library")
}

android {
  namespace = "dev.orbis"
  compileSdk = 36
  defaultConfig {
    minSdk = 24
  }
  sourceSets["main"].kotlin.srcDir("src/generated/kotlin")
}
