#ifndef OrbConstraints_D_H
#define OrbConstraints_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef struct OrbConstraints {
  double min_zoom;
  double max_zoom;
  double min_pitch;
  double max_pitch;
  bool bounded;
  double west;
  double south;
  double east;
  double north;
} OrbConstraints;

typedef struct OrbConstraints_option {union { OrbConstraints ok; }; bool is_ok; } OrbConstraints_option;



#endif // OrbConstraints_D_H
