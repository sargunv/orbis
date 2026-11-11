# liborbis.so calls rustls-platform-verifier's Kotlin half through JNI,
# which R8 can't see.
-keep, includedescriptorclasses class org.rustls.platformverifier.** { *; }
