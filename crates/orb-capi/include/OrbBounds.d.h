#ifndef OrbBounds_D_H
#define OrbBounds_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef struct OrbBounds {
  double west;
  double south;
  double east;
  double north;
} OrbBounds;

typedef struct OrbBounds_option {union { OrbBounds ok; }; bool is_ok; } OrbBounds_option;



#endif // OrbBounds_D_H
