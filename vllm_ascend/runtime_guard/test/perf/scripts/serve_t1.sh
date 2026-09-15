#!/usr/bin/env bash
# T1: product HEAD, plain serve (reload=0 default, all detectors off).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
log "[T1] serve product HEAD (reload=0, detectors off) from $PRODUCT_ROOT"
serve_and_wait "$PRODUCT_ROOT" "t1"
