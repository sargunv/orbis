#ifndef OrbBackend_D_H
#define OrbBackend_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef enum OrbBackend {
  OrbBackend_Auto = 0,
  OrbBackend_Modern = 1,
  OrbBackend_Compat = 2,
} OrbBackend;

typedef struct OrbBackend_option {union { OrbBackend ok; }; bool is_ok; } OrbBackend_option;



#endif // OrbBackend_D_H
