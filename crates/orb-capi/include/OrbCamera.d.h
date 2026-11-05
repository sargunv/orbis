#ifndef OrbCamera_D_H
#define OrbCamera_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"

#include "OrbPadding.d.h"




typedef struct OrbCamera {
  double lon;
  double lat;
  double zoom;
  double bearing;
  double pitch;
  double roll;
  double elevation;
  double fov;
  OrbPadding padding;
} OrbCamera;

typedef struct OrbCamera_option {union { OrbCamera ok; }; bool is_ok; } OrbCamera_option;



#endif // OrbCamera_D_H
