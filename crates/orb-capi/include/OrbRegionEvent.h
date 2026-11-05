#ifndef OrbRegionEvent_H
#define OrbRegionEvent_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"

#include "OrbRegionEventKind.d.h"

#include "OrbRegionEvent.d.h"






OrbRegionEventKind OrbRegionEvent_kind(const OrbRegionEvent* self);

uint32_t OrbRegionEvent_region(const OrbRegionEvent* self);

uint64_t OrbRegionEvent_done(const OrbRegionEvent* self);

uint64_t OrbRegionEvent_total(const OrbRegionEvent* self);

uint64_t OrbRegionEvent_bytes(const OrbRegionEvent* self);

uint64_t OrbRegionEvent_failed(const OrbRegionEvent* self);

void OrbRegionEvent_message(const OrbRegionEvent* self, DiplomatWrite* write);

void OrbRegionEvent_destroy(OrbRegionEvent* self);





#endif // OrbRegionEvent_H
