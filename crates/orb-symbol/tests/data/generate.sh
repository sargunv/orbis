#!/bin/sh
# Regenerates the GL JS golden vectors: generate.sh <gl-js checkout at the pin>
# See README.md.
set -eu
gljs=$(cd "$1" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$gljs/orbis"
cp "$here/generate.ts" "$gljs/orbis/generate.ts"
(cd "$gljs" && ./node_modules/.bin/rolldown orbis/generate.ts --platform node --format esm -o orbis/generate.mjs > /dev/null)
node "$gljs/orbis/generate.mjs" "$here"
