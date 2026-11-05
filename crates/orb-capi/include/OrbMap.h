#ifndef OrbMap_H
#define OrbMap_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"

#include "OrbBounds.d.h"
#include "OrbButton.d.h"
#include "OrbCamera.d.h"
#include "OrbCameraTarget.d.h"
#include "OrbError.d.h"
#include "OrbEvent.d.h"
#include "OrbFeature.d.h"
#include "OrbFeel.d.h"
#include "OrbFrame.d.h"
#include "OrbKey.d.h"
#include "OrbLngLat.d.h"
#include "OrbModifiers.d.h"
#include "OrbPoint.d.h"
#include "OrbPointerKind.d.h"
#include "OrbRuntime.d.h"
#include "OrbTarget.d.h"
#include "OrbWheelUnit.d.h"

#include "OrbMap.d.h"





typedef struct DiplomatCallback_OrbMap_set_wake_wake {
    const void* data;
    void (*run_callback)(const void*);
    void (*destructor)(const void*);
} DiplomatCallback_OrbMap_set_wake_wake;

OrbMap* OrbMap_create(const OrbRuntime* runtime, OrbFeel feel, double clear_r, double clear_g, double clear_b, double clear_a);

void OrbMap_set_wake(OrbMap* self, DiplomatCallback_OrbMap_set_wake_wake wake_cb_wrap);

bool OrbMap_needs_frame(const OrbMap* self);

void OrbMap_load_style_url(OrbMap* self, DiplomatStringView url);

typedef struct OrbMap_set_style_json_result {union { OrbError err;}; bool is_ok;} OrbMap_set_style_json_result;
OrbMap_set_style_json_result OrbMap_set_style_json(OrbMap* self, DiplomatStringView json);

typedef struct OrbMap_edit_result {union {uint64_t ok; OrbError err;}; bool is_ok;} OrbMap_edit_result;
OrbMap_edit_result OrbMap_edit(OrbMap* self, DiplomatStringView commands);

typedef struct OrbMap_add_image_result {union { OrbError err;}; bool is_ok;} OrbMap_add_image_result;
OrbMap_add_image_result OrbMap_add_image(OrbMap* self, DiplomatStringView name, uint32_t width, uint32_t height, DiplomatU8View pixels, DiplomatStringView options);

typedef struct OrbMap_update_image_result {union { OrbError err;}; bool is_ok;} OrbMap_update_image_result;
OrbMap_update_image_result OrbMap_update_image(OrbMap* self, DiplomatStringView name, uint32_t width, uint32_t height, DiplomatU8View pixels);

typedef struct OrbMap_remove_image_result {union { OrbError err;}; bool is_ok;} OrbMap_remove_image_result;
OrbMap_remove_image_result OrbMap_remove_image(OrbMap* self, DiplomatStringView name);

bool OrbMap_has_image(const OrbMap* self, DiplomatStringView name);

uint64_t OrbMap_generation(const OrbMap* self);

OrbCamera OrbMap_camera(const OrbMap* self);

void OrbMap_set_center_clamped_to_ground(OrbMap* self, bool clamped);

bool OrbMap_center_clamped_to_ground(const OrbMap* self);

void OrbMap_set_center_elevation(OrbMap* self, double elevation);

void OrbMap_jump_to(OrbMap* self, OrbCameraTarget target);

uint64_t OrbMap_ease_to(OrbMap* self, OrbCameraTarget target, double duration_ms);

uint64_t OrbMap_fly_to(OrbMap* self, OrbCameraTarget target, double duration_ms);

uint64_t OrbMap_fit_bounds(OrbMap* self, OrbBounds bounds, double padding, double duration_ms);

void OrbMap_stop(OrbMap* self);

bool OrbMap_pointer_down(OrbMap* self, uint32_t id, OrbPointerKind kind, OrbButton button, double x, double y, OrbModifiers modifiers);

bool OrbMap_pointer_move(OrbMap* self, uint32_t id, double x, double y);

bool OrbMap_pointer_up(OrbMap* self, uint32_t id, double x, double y);

bool OrbMap_pointer_cancel(OrbMap* self, uint32_t id);

bool OrbMap_wheel(OrbMap* self, double x, double y, double dx, double dy, OrbWheelUnit unit, OrbModifiers modifiers);

bool OrbMap_key(OrbMap* self, OrbKey key, OrbModifiers modifiers);

bool OrbMap_blur(OrbMap* self);

OrbFeature* OrbMap_hit_test(const OrbMap* self, double x, double y);

uint64_t OrbMap_query_rendered(OrbMap* self, double x0, double y0, double x1, double y1);

OrbPoint OrbMap_project(const OrbMap* self, double lon, double lat);

OrbLngLat OrbMap_unproject(const OrbMap* self, double x, double y);

typedef struct OrbMap_render_result {union {OrbFrame ok; OrbError err;}; bool is_ok;} OrbMap_render_result;
OrbMap_render_result OrbMap_render(OrbMap* self, OrbTarget* target);

OrbEvent* OrbMap_poll_event(OrbMap* self);

void OrbMap_take_warnings(OrbMap* self, DiplomatWrite* write);

void OrbMap_destroy(OrbMap* self);





#endif // OrbMap_H
