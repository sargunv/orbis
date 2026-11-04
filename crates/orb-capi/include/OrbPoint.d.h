#ifndef OrbPoint_D_H
#define OrbPoint_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef struct OrbPoint {
  double x;
  double y;
  bool valid;
} OrbPoint;

typedef struct OrbPoint_option {union { OrbPoint ok; }; bool is_ok; } OrbPoint_option;



#endif // OrbPoint_D_H
