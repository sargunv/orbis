#ifndef OrbFeature_H
#define OrbFeature_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"


#include "OrbFeature.d.h"






void OrbFeature_layer(const OrbFeature* self, DiplomatWrite* write);

void OrbFeature_source(const OrbFeature* self, DiplomatWrite* write);

void OrbFeature_source_layer(const OrbFeature* self, DiplomatWrite* write);

bool OrbFeature_has_id(const OrbFeature* self);

void OrbFeature_id(const OrbFeature* self, DiplomatWrite* write);

uint64_t OrbFeature_generation(const OrbFeature* self);

void OrbFeature_destroy(OrbFeature* self);





#endif // OrbFeature_H
