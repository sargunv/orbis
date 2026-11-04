#ifndef OrbButton_D_H
#define OrbButton_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef enum OrbButton {
  OrbButton_Primary = 0,
  OrbButton_Secondary = 1,
  OrbButton_Middle = 2,
} OrbButton;

typedef struct OrbButton_option {union { OrbButton ok; }; bool is_ok; } OrbButton_option;



#endif // OrbButton_D_H
