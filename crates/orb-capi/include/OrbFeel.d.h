#ifndef OrbFeel_D_H
#define OrbFeel_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef enum OrbFeel {
  OrbFeel_Desktop = 0,
  OrbFeel_Ios = 1,
  OrbFeel_Android = 2,
} OrbFeel;

typedef struct OrbFeel_option {union { OrbFeel ok; }; bool is_ok; } OrbFeel_option;



#endif // OrbFeel_D_H
