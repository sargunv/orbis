#ifndef OrbPointerKind_D_H
#define OrbPointerKind_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef enum OrbPointerKind {
  OrbPointerKind_Mouse = 0,
  OrbPointerKind_Touch = 1,
  OrbPointerKind_Pen = 2,
} OrbPointerKind;

typedef struct OrbPointerKind_option {union { OrbPointerKind ok; }; bool is_ok; } OrbPointerKind_option;



#endif // OrbPointerKind_D_H
