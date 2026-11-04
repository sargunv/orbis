#ifndef OrbModifiers_D_H
#define OrbModifiers_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef struct OrbModifiers {
  bool shift;
  bool ctrl;
  bool alt;
  bool meta;
} OrbModifiers;

typedef struct OrbModifiers_option {union { OrbModifiers ok; }; bool is_ok; } OrbModifiers_option;



#endif // OrbModifiers_D_H
