#ifndef OrbLngLat_D_H
#define OrbLngLat_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef struct OrbLngLat {
  double lon;
  double lat;
  bool valid;
} OrbLngLat;

typedef struct OrbLngLat_option {union { OrbLngLat ok; }; bool is_ok; } OrbLngLat_option;



#endif // OrbLngLat_D_H
