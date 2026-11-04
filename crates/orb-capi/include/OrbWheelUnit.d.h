#ifndef OrbWheelUnit_D_H
#define OrbWheelUnit_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef enum OrbWheelUnit {
  OrbWheelUnit_Pixel = 0,
  OrbWheelUnit_Line = 1,
  OrbWheelUnit_Page = 2,
} OrbWheelUnit;

typedef struct OrbWheelUnit_option {union { OrbWheelUnit ok; }; bool is_ok; } OrbWheelUnit_option;



#endif // OrbWheelUnit_D_H
