#!/bin/sh
set -eu
# Mesa's filename varies by distribution/architecture. Restrict the Vulkan
# loader to this software ICD so a mounted host GPU can never change the run.
set -- /usr/share/vulkan/icd.d/lvp_icd*.json "$@"
export VK_DRIVER_FILES="$1"
shift
exec "$@"
