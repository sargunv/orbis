#ifndef OrbEvent_H
#define OrbEvent_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"

#include "OrbButton.d.h"
#include "OrbChange.d.h"
#include "OrbEventKind.d.h"
#include "OrbPhase.d.h"

#include "OrbEvent.d.h"






OrbEventKind OrbEvent_kind(const OrbEvent* self);

double OrbEvent_x(const OrbEvent* self);

double OrbEvent_y(const OrbEvent* self);

uint64_t OrbEvent_generation(const OrbEvent* self);

uint64_t OrbEvent_id(const OrbEvent* self);

bool OrbEvent_flag(const OrbEvent* self);

OrbChange OrbEvent_change(const OrbEvent* self);

OrbPhase OrbEvent_phase(const OrbEvent* self);

OrbButton OrbEvent_button(const OrbEvent* self);

void OrbEvent_text(const OrbEvent* self, DiplomatWrite* write);

void OrbEvent_destroy(OrbEvent* self);





#endif // OrbEvent_H
