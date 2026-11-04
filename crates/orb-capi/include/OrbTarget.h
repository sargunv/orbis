#ifndef OrbTarget_H
#define OrbTarget_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"

#include "OrbError.d.h"
#include "OrbRuntime.d.h"

#include "OrbTarget.d.h"






OrbTarget* OrbTarget_create_texture(const OrbRuntime* runtime, uint32_t width, uint32_t height, double scale);

typedef struct OrbTarget_create_metal_layer_result {union {OrbTarget* ok; OrbError err;}; bool is_ok;} OrbTarget_create_metal_layer_result;
OrbTarget_create_metal_layer_result OrbTarget_create_metal_layer(const OrbRuntime* runtime, size_t layer, uint32_t width, uint32_t height, double scale);

typedef struct OrbTarget_create_android_window_result {union {OrbTarget* ok; OrbError err;}; bool is_ok;} OrbTarget_create_android_window_result;
OrbTarget_create_android_window_result OrbTarget_create_android_window(const OrbRuntime* runtime, size_t window, uint32_t width, uint32_t height, double scale);

void OrbTarget_resize(OrbTarget* self, uint32_t width, uint32_t height, double scale);

uint32_t OrbTarget_width(const OrbTarget* self);

uint32_t OrbTarget_height(const OrbTarget* self);

typedef struct OrbTarget_read_pixels_result {union { OrbError err;}; bool is_ok;} OrbTarget_read_pixels_result;
OrbTarget_read_pixels_result OrbTarget_read_pixels(OrbTarget* self);

DiplomatU8View OrbTarget_pixels(const OrbTarget* self);

void OrbTarget_destroy(OrbTarget* self);





#endif // OrbTarget_H
