#ifndef OrbPadding_D_H
#define OrbPadding_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef struct OrbPadding {
  double top;
  double right;
  double bottom;
  double left;
} OrbPadding;

typedef struct OrbPadding_option {union { OrbPadding ok; }; bool is_ok; } OrbPadding_option;



#endif // OrbPadding_D_H
