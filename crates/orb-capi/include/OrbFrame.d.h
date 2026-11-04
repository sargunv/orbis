#ifndef OrbFrame_D_H
#define OrbFrame_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef struct OrbFrame {
  bool drawn;
  uint64_t generation;
  bool idle;
  bool loaded;
  uint32_t tiles_drawn;
  uint32_t tiles_loading;
  uint32_t draws;
  uint32_t dispatches;
} OrbFrame;

typedef struct OrbFrame_option {union { OrbFrame ok; }; bool is_ok; } OrbFrame_option;



#endif // OrbFrame_D_H
