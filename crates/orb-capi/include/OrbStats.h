#ifndef OrbStats_H
#define OrbStats_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"


#include "OrbStats.d.h"






bool OrbStats_compat(const OrbStats* self);

void OrbStats_selection(const OrbStats* self, DiplomatWrite* write);

uint32_t OrbStats_pipelines_created(const OrbStats* self);

double OrbStats_pipeline_ms(const OrbStats* self);

void OrbStats_pipeline_timings(const OrbStats* self, DiplomatWrite* write);

uint64_t OrbStats_http_cache_bytes(const OrbStats* self);

void OrbStats_destroy(OrbStats* self);





#endif // OrbStats_H
