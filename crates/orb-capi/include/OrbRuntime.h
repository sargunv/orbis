#ifndef OrbRuntime_H
#define OrbRuntime_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"

#include "OrbError.d.h"

#include "OrbRuntime.d.h"






typedef struct OrbRuntime_create_result {union {OrbRuntime* ok; OrbError err;}; bool is_ok;} OrbRuntime_create_result;
OrbRuntime_create_result OrbRuntime_create(uint32_t job_threads);

uint32_t OrbRuntime_abi_version(void);

uint32_t OrbRuntime_pipelines_created(const OrbRuntime* self);

void OrbRuntime_destroy(OrbRuntime* self);





#endif // OrbRuntime_H
