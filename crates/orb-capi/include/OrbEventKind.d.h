#ifndef OrbEventKind_D_H
#define OrbEventKind_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef enum OrbEventKind {
  OrbEventKind_StyleLoad = 0,
  OrbEventKind_StyleData = 1,
  OrbEventKind_SourceData = 2,
  OrbEventKind_Load = 3,
  OrbEventKind_Idle = 4,
  OrbEventKind_Error = 5,
  OrbEventKind_Camera = 6,
  OrbEventKind_AnimationEnd = 7,
  OrbEventKind_Click = 8,
  OrbEventKind_DoubleClick = 9,
  OrbEventKind_QueryResult = 10,
  OrbEventKind_Capability = 11,
  OrbEventKind_StyleImageMissing = 12,
  OrbEventKind_VideoOpen = 13,
  OrbEventKind_VideoPlay = 14,
  OrbEventKind_VideoPause = 15,
  OrbEventKind_VideoSeek = 16,
  OrbEventKind_VideoClose = 17,
} OrbEventKind;

typedef struct OrbEventKind_option {union { OrbEventKind ok; }; bool is_ok; } OrbEventKind_option;



#endif // OrbEventKind_D_H
