#ifndef OrbKey_D_H
#define OrbKey_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef enum OrbKey {
  OrbKey_Left = 0,
  OrbKey_Right = 1,
  OrbKey_Up = 2,
  OrbKey_Down = 3,
  OrbKey_ZoomIn = 4,
  OrbKey_ZoomOut = 5,
} OrbKey;

typedef struct OrbKey_option {union { OrbKey ok; }; bool is_ok; } OrbKey_option;



#endif // OrbKey_D_H
