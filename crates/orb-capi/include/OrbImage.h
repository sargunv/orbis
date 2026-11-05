#ifndef OrbImage_H
#define OrbImage_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"


#include "OrbImage.d.h"






uint32_t OrbImage_width(const OrbImage* self);

uint32_t OrbImage_height(const OrbImage* self);

DiplomatU8View OrbImage_pixels(const OrbImage* self);

void OrbImage_destroy(OrbImage* self);





#endif // OrbImage_H
