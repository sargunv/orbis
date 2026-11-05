#ifndef OrbTile_D_H
#define OrbTile_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef struct OrbTile {
  uint8_t z;
  uint32_t x;
  uint32_t y;
} OrbTile;

typedef struct OrbTile_option {union { OrbTile ok; }; bool is_ok; } OrbTile_option;



#endif // OrbTile_D_H
