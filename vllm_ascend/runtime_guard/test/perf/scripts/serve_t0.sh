#!/usr/bin/env bash
# T0: merge-base 37e382498 worktree (no RuntimeGuardProcessor.bind). Serve + wait health.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
log "[T0] serve merge-base (no runtime_guard) from $T0_ROOT"
serve_and_wait "$T0_ROOT" "t0"
