/* The Android side the C ABI can't reach: a Surface's ANativeWindow, which
 * only JNI can get. Linked into liborbis.so with the generated JNI shim. */

#include <android/native_window_jni.h>
#include <jni.h>
#include <stdint.h>

JNIEXPORT jlong JNICALL Java_dev_orbis_android_NativeWindows_fromSurface(
  JNIEnv* env, jclass cls, jobject surface
) {
  (void)cls;
  return (jlong)(intptr_t)ANativeWindow_fromSurface(env, surface);
}

JNIEXPORT void JNICALL Java_dev_orbis_android_NativeWindows_release(
  JNIEnv* env, jclass cls, jlong window
) {
  (void)env;
  (void)cls;
  ANativeWindow_release((ANativeWindow*)(intptr_t)window);
}
