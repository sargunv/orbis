#ifndef OrbRuntime_H
#define OrbRuntime_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"

#include "OrbBackend.d.h"
#include "OrbError.d.h"
#include "OrbRegionEvent.d.h"
#include "OrbStats.d.h"

#include "OrbRuntime.d.h"






typedef struct OrbRuntime_create_result {union {OrbRuntime* ok; OrbError err;}; bool is_ok;} OrbRuntime_create_result;
OrbRuntime_create_result OrbRuntime_create(uint32_t job_threads);

typedef struct OrbRuntime_create_with_result {union {OrbRuntime* ok; OrbError err;}; bool is_ok;} OrbRuntime_create_with_result;
OrbRuntime_create_with_result OrbRuntime_create_with(uint32_t job_threads, OrbBackend backend, DiplomatStringView state_dir, uint32_t android_api_level);

OrbStats* OrbRuntime_stats(const OrbRuntime* self);

void OrbRuntime_lose_device(OrbRuntime* self);

uint32_t OrbRuntime_abi_version(void);

typedef struct OrbRuntime_download_region_result {union {uint32_t ok; OrbError err;}; bool is_ok;} OrbRuntime_download_region_result;
OrbRuntime_download_region_result OrbRuntime_download_region(OrbRuntime* self, DiplomatStringView name, DiplomatStringView style_url, double west, double south, double east, double north, double min_zoom, double max_zoom);

typedef struct OrbRuntime_delete_region_result {union { OrbError err;}; bool is_ok;} OrbRuntime_delete_region_result;
OrbRuntime_delete_region_result OrbRuntime_delete_region(OrbRuntime* self, uint32_t id);

typedef struct OrbRuntime_regions_result {union { OrbError err;}; bool is_ok;} OrbRuntime_regions_result;
OrbRuntime_regions_result OrbRuntime_regions(const OrbRuntime* self, DiplomatWrite* write);

OrbRegionEvent* OrbRuntime_poll_region_event(OrbRuntime* self);

void OrbRuntime_clear_http_cache(OrbRuntime* self);

void OrbRuntime_destroy(OrbRuntime* self);





#endif // OrbRuntime_H
