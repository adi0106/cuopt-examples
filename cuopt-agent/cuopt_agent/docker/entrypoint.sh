#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Maps the container-level DEBUG_LEVEL flag onto LOG_LEVEL, which the NAT
# config files reference via ${LOG_LEVEL:-WARNING}, and onto the nat CLI's
# --log-level option. The YAML setting only sets the console handler's level;
# the CLI flag sets the root/nat logger level. Both are needed — otherwise
# DEBUG records are filtered at the logger before reaching the handler.
#
#   DEBUG_LEVEL=0  -> LOG_LEVEL=WARNING  (quiet; no debug output)
#   DEBUG_LEVEL=1  -> LOG_LEVEL=DEBUG    (verbose; full logs + debug info)

set -e

case "${DEBUG_LEVEL:-0}" in
    0) export LOG_LEVEL=WARNING ;;
    1) export LOG_LEVEL=DEBUG
       export PYTHONUNBUFFERED=1
       echo "[entrypoint] DEBUG_LEVEL=1 -> LOG_LEVEL=DEBUG" >&2
       ;;
    *) echo "[entrypoint] invalid DEBUG_LEVEL='${DEBUG_LEVEL}', expected 0 or 1" >&2
       exit 2
       ;;
esac

if [ "$1" = "nat" ]; then
    shift
    # Only `nat serve` needs the nginx front (Accept-header routing for
    # /generate). `nat eval` and other subcommands run NAT directly.
    # nginx runs as a backgrounded child; when NAT (PID 1 after exec) exits,
    # docker tears down the cgroup and nginx goes with it.
    if [ "$1" = "serve" ]; then
        nginx -g 'daemon off;' &
        echo "[entrypoint] nginx started (pid=$!) on :8000 -> NAT 127.0.0.1:8001" >&2
    fi
    exec nat --log-level "${LOG_LEVEL}" "$@"
fi

exec "$@"
