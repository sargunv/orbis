#ifndef OrbChange_D_H
#define OrbChange_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef enum OrbChange {
  OrbChange_Move = 0,
  OrbChange_Zoom = 1,
  OrbChange_Rotate = 2,
  OrbChange_Pitch = 3,
} OrbChange;

typedef struct OrbChange_option {union { OrbChange ok; }; bool is_ok; } OrbChange_option;



#endif // OrbChange_D_H
