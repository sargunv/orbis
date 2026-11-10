package dev.orbis.android

import android.view.Surface

/** A Surface's `ANativeWindow`, held until released. */
internal object NativeWindows {
  init {
    System.loadLibrary("orbis")
  }

  @JvmStatic external fun fromSurface(surface: Surface): Long

  @JvmStatic external fun release(window: Long)
}
