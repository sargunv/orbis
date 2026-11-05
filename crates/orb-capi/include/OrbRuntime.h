#ifndef OrbRuntime_H
#define OrbRuntime_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"

#include "OrbBackend.d.h"
#include "OrbError.d.h"

#include "OrbRuntime.d.h"






typedef struct OrbRuntime_create_result {union {OrbRuntime* ok; OrbError err;}; bool is_ok;} OrbRuntime_create_result;
OrbRuntime_create_result OrbRuntime_create(uint32_t job_threads);

typedef struct OrbRuntime_create_with_result {union {OrbRuntime* ok; OrbError err;}; bool is_ok;} OrbRuntime_create_with_result;
OrbRuntime_create_with_result OrbRuntime_create_with(uint32_t job_threads, OrbBackend backend, DiplomatStringView state_dir, uint32_t android_api_level);

bool OrbRuntime_compat(const OrbRuntime* self);

void OrbRuntime_selection(const OrbRuntime* self, DiplomatWrite* write);

void OrbRuntime_lose_device(OrbRuntime* self);

typedef struct OrbRuntime_recover_result {union { OrbError err;}; bool is_ok;} OrbRuntime_recover_result;
OrbRuntime_recover_result OrbRuntime_recover(OrbRuntime* self);

uint32_t OrbRuntime_abi_version(void);

uint32_t OrbRuntime_pipelines_created(const OrbRuntime* self);

typedef struct OrbRuntime_set_pipeline_cache_dir_result {union { OrbError err;}; bool is_ok;} OrbRuntime_set_pipeline_cache_dir_result;
OrbRuntime_set_pipeline_cache_dir_result OrbRuntime_set_pipeline_cache_dir(OrbRuntime* self, DiplomatStringView dir);

double OrbRuntime_pipeline_ms(const OrbRuntime* self);

void OrbRuntime_destroy(OrbRuntime* self);





#endif // OrbRuntime_H
