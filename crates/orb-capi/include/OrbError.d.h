#ifndef OrbError_D_H
#define OrbError_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef enum OrbError {
  OrbError_InvalidArgument = 1,
  OrbError_InvalidState = 2,
  OrbError_Unsupported = 3,
  OrbError_Native = 4,
  OrbError_TargetLost = 5,
} OrbError;

typedef struct OrbError_option {union { OrbError ok; }; bool is_ok; } OrbError_option;



#endif // OrbError_D_H
