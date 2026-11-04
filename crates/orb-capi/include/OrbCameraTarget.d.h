#ifndef OrbCameraTarget_D_H
#define OrbCameraTarget_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef struct OrbCameraTarget {
  double lon;
  double lat;
  double zoom;
  double bearing;
  double pitch;
} OrbCameraTarget;

typedef struct OrbCameraTarget_option {union { OrbCameraTarget ok; }; bool is_ok; } OrbCameraTarget_option;



#endif // OrbCameraTarget_D_H
