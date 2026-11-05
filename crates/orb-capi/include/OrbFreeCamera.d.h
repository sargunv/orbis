#ifndef OrbFreeCamera_D_H
#define OrbFreeCamera_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef struct OrbFreeCamera {
  double lon;
  double lat;
  double altitude;
  double bearing;
  double pitch;
  double roll;
} OrbFreeCamera;

typedef struct OrbFreeCamera_option {union { OrbFreeCamera ok; }; bool is_ok; } OrbFreeCamera_option;



#endif // OrbFreeCamera_D_H
