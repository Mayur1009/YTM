#!/usr/bin/env bash
args=()
for a in "$@"; do [[ "$a" == "-mrdrnd" ]] || args+=("$a"); done
exec clang -fuse-ld=/usr/bin/ld "${args[@]}"
