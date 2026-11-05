#ifndef OrbRegionEventKind_D_H
#define OrbRegionEventKind_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef enum OrbRegionEventKind {
  OrbRegionEventKind_Progress = 0,
  OrbRegionEventKind_Complete = 1,
  OrbRegionEventKind_Failed = 2,
} OrbRegionEventKind;

typedef struct OrbRegionEventKind_option {union { OrbRegionEventKind ok; }; bool is_ok; } OrbRegionEventKind_option;



#endif // OrbRegionEventKind_D_H
