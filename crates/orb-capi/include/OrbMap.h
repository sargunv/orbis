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
#include "OrbConstraints.d.h"
#include "OrbError.d.h"
#include "OrbEvent.d.h"
#include "OrbFeature.d.h"
#include "OrbFeel.d.h"
#include "OrbFrame.d.h"
#include "OrbFreeCamera.d.h"
#include "OrbImage.d.h"
#include "OrbKey.d.h"
#include "OrbLngLat.d.h"
#include "OrbModifiers.d.h"
#include "OrbPoint.d.h"
#include "OrbPointerKind.d.h"
#include "OrbRuntime.d.h"
#include "OrbTarget.d.h"
#include "OrbTile.d.h"
#include "OrbTransaction.d.h"
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

uint64_t OrbMap_add_marker(OrbMap* self, double lon, double lat, DiplomatStringView image, DiplomatStringView anchor);

typedef struct OrbMap_update_marker_result {union { OrbError err;}; bool is_ok;} OrbMap_update_marker_result;
OrbMap_update_marker_result OrbMap_update_marker(OrbMap* self, uint64_t id, double lon, double lat, DiplomatStringView image, DiplomatStringView anchor);

typedef struct OrbMap_remove_marker_result {union { OrbError err;}; bool is_ok;} OrbMap_remove_marker_result;
OrbMap_remove_marker_result OrbMap_remove_marker(OrbMap* self, uint64_t id);

void OrbMap_set_location(OrbMap* self, double lon, double lat, double accuracy, double heading);

void OrbMap_clear_location(OrbMap* self);

void OrbMap_retry_failed_tiles(OrbMap* self);

void OrbMap_load_style_url(OrbMap* self, DiplomatStringView url);

typedef struct OrbMap_set_style_json_result {union { OrbError err;}; bool is_ok;} OrbMap_set_style_json_result;
OrbMap_set_style_json_result OrbMap_set_style_json(OrbMap* self, DiplomatStringView json);

typedef struct OrbMap_edit_result {union {uint64_t ok; OrbError err;}; bool is_ok;} OrbMap_edit_result;
OrbMap_edit_result OrbMap_edit(OrbMap* self, const OrbTransaction* transaction);

void OrbMap_set_plays_video(OrbMap* self, bool plays);

typedef struct OrbMap_source_frame_result {union { OrbError err;}; bool is_ok;} OrbMap_source_frame_result;
OrbMap_source_frame_result OrbMap_source_frame(OrbMap* self, DiplomatStringView id, uint32_t width, uint32_t height, DiplomatU8View pixels);

typedef struct OrbMap_host_tile_result {union { OrbError err;}; bool is_ok;} OrbMap_host_tile_result;
OrbMap_host_tile_result OrbMap_host_tile(OrbMap* self, DiplomatStringView source, OrbTile tile, DiplomatU8View bytes);

typedef struct OrbMap_host_tile_geojson_result {union { OrbError err;}; bool is_ok;} OrbMap_host_tile_geojson_result;
OrbMap_host_tile_geojson_result OrbMap_host_tile_geojson(OrbMap* self, DiplomatStringView source, OrbTile tile, DiplomatStringView layers);

typedef struct OrbMap_host_tile_empty_result {union { OrbError err;}; bool is_ok;} OrbMap_host_tile_empty_result;
OrbMap_host_tile_empty_result OrbMap_host_tile_empty(OrbMap* self, DiplomatStringView source, OrbTile tile);

typedef struct OrbMap_reload_host_tiles_result {union { OrbError err;}; bool is_ok;} OrbMap_reload_host_tiles_result;
OrbMap_reload_host_tiles_result OrbMap_reload_host_tiles(OrbMap* self, DiplomatStringView source);

typedef struct OrbMap_video_failed_result {union { OrbError err;}; bool is_ok;} OrbMap_video_failed_result;
OrbMap_video_failed_result OrbMap_video_failed(OrbMap* self, DiplomatStringView id, DiplomatStringView message);

typedef struct OrbMap_play_video_result {union { OrbError err;}; bool is_ok;} OrbMap_play_video_result;
OrbMap_play_video_result OrbMap_play_video(OrbMap* self, DiplomatStringView id);

typedef struct OrbMap_pause_video_result {union { OrbError err;}; bool is_ok;} OrbMap_pause_video_result;
OrbMap_pause_video_result OrbMap_pause_video(OrbMap* self, DiplomatStringView id);

typedef struct OrbMap_seek_video_result {union { OrbError err;}; bool is_ok;} OrbMap_seek_video_result;
OrbMap_seek_video_result OrbMap_seek_video(OrbMap* self, DiplomatStringView id, double seconds);

typedef struct OrbMap_add_image_result {union { OrbError err;}; bool is_ok;} OrbMap_add_image_result;
OrbMap_add_image_result OrbMap_add_image(OrbMap* self, DiplomatStringView name, uint32_t width, uint32_t height, DiplomatU8View pixels, DiplomatStringView options);

typedef struct OrbMap_update_image_result {union { OrbError err;}; bool is_ok;} OrbMap_update_image_result;
OrbMap_update_image_result OrbMap_update_image(OrbMap* self, DiplomatStringView name, uint32_t width, uint32_t height, DiplomatU8View pixels);

typedef struct OrbMap_remove_image_result {union { OrbError err;}; bool is_ok;} OrbMap_remove_image_result;
OrbMap_remove_image_result OrbMap_remove_image(OrbMap* self, DiplomatStringView name);

bool OrbMap_has_image(const OrbMap* self, DiplomatStringView name);

uint64_t OrbMap_generation(const OrbMap* self);

OrbCamera OrbMap_camera(const OrbMap* self);

OrbFreeCamera OrbMap_free_camera(const OrbMap* self);

void OrbMap_jump_to_free(OrbMap* self, OrbFreeCamera camera);

void OrbMap_set_center_clamped_to_ground(OrbMap* self, bool clamped);

bool OrbMap_center_clamped_to_ground(const OrbMap* self);

void OrbMap_set_center_elevation(OrbMap* self, double elevation);

void OrbMap_jump_to(OrbMap* self, OrbCameraTarget target);

uint64_t OrbMap_ease_to(OrbMap* self, OrbCameraTarget target, double duration_ms);

uint64_t OrbMap_fly_to(OrbMap* self, OrbCameraTarget target, double duration_ms);

uint64_t OrbMap_fit_bounds(OrbMap* self, OrbBounds bounds, double padding, double duration_ms);

void OrbMap_stop(OrbMap* self);

void OrbMap_set_constraints(OrbMap* self, OrbConstraints constraints);

void OrbMap_semantic_snapshot(const OrbMap* self, DiplomatWrite* write);

bool OrbMap_pointer_down(OrbMap* self, uint32_t id, OrbPointerKind kind, OrbButton button, double x, double y, OrbModifiers modifiers, double time_ms);

bool OrbMap_pointer_move(OrbMap* self, uint32_t id, double x, double y, double time_ms);

bool OrbMap_pointer_up(OrbMap* self, uint32_t id, double x, double y, double time_ms);

bool OrbMap_pointer_cancel(OrbMap* self, uint32_t id, double time_ms);

bool OrbMap_wheel(OrbMap* self, double x, double y, double dx, double dy, OrbWheelUnit unit, OrbModifiers modifiers, double time_ms);

bool OrbMap_key(OrbMap* self, OrbKey key, OrbModifiers modifiers, double time_ms);

bool OrbMap_blur(OrbMap* self, double time_ms);

OrbFeature* OrbMap_hit_test(const OrbMap* self, double x, double y);

uint64_t OrbMap_query_rendered(OrbMap* self, double x0, double y0, double x1, double y1);

uint64_t OrbMap_query_source(OrbMap* self, DiplomatStringView source, DiplomatStringView source_layer, DiplomatStringView filter);

uint64_t OrbMap_cluster_children(OrbMap* self, DiplomatStringView source, uint64_t cluster);

uint64_t OrbMap_cluster_leaves(OrbMap* self, DiplomatStringView source, uint64_t cluster, uint32_t limit, uint32_t offset);

uint64_t OrbMap_cluster_expansion_zoom(OrbMap* self, DiplomatStringView source, uint64_t cluster);

OrbPoint OrbMap_project(const OrbMap* self, double lon, double lat);

OrbPoint OrbMap_project_with(const OrbMap* self, OrbCameraTarget camera, double lon, double lat);

OrbLngLat OrbMap_unproject_with(const OrbMap* self, OrbCameraTarget camera, double x, double y);

OrbLngLat OrbMap_unproject(const OrbMap* self, double x, double y);

typedef struct OrbMap_snapshot_result {union {OrbImage* ok; OrbError err;}; bool is_ok;} OrbMap_snapshot_result;
OrbMap_snapshot_result OrbMap_snapshot(OrbMap* self, uint32_t width, uint32_t height, double scale);

typedef struct OrbMap_render_tile_result {union {OrbImage* ok; OrbError err;}; bool is_ok;} OrbMap_render_tile_result;
OrbMap_render_tile_result OrbMap_render_tile(OrbMap* self, DiplomatStringView tile_matrix_set, uint8_t z, uint32_t col, uint32_t row, uint32_t size, double scale);

typedef struct OrbMap_render_result {union {OrbFrame ok; OrbError err;}; bool is_ok;} OrbMap_render_result;
OrbMap_render_result OrbMap_render(OrbMap* self, OrbTarget* target, double frame_time_ms);

OrbEvent* OrbMap_poll_event(OrbMap* self);

void OrbMap_destroy(OrbMap* self);





#endif // OrbMap_H
