#ifndef OrbPhase_D_H
#define OrbPhase_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef enum OrbPhase {
  OrbPhase_Start = 0,
  OrbPhase_Progress = 1,
  OrbPhase_End = 2,
} OrbPhase;

typedef struct OrbPhase_option {union { OrbPhase ok; }; bool is_ok; } OrbPhase_option;



#endif // OrbPhase_D_H
