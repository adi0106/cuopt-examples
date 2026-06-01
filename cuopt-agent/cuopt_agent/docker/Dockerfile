# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Production container for NeMo Agent Toolkit with cuOpt skills.
# Build context must be repo root (cuopt-agent/).
#
# docker build -f cuopt_agent/docker/Dockerfile -t cuopt-agent .

FROM nvcr.io/nvidia/cuopt/cuopt:26.2.0-cuda12.9-py3.13

# nginx + njs (JavaScript body filter). Ubuntu 22.04 ships nginx 1.18 without
# the njs module, so we pull both from the official nginx.org repo. The
# load_module directive must live in main context, so we prepend it to
# /etc/nginx/nginx.conf at build time.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates gnupg && \
    curl -fsSL https://nginx.org/keys/nginx_signing.key | gpg --dearmor -o /usr/share/keyrings/nginx-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/nginx-archive-keyring.gpg] https://nginx.org/packages/ubuntu/ jammy nginx" > /etc/apt/sources.list.d/nginx.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends nginx nginx-module-njs && \
    sed -i '1i load_module modules/ngx_http_js_module.so;' /etc/nginx/nginx.conf && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# ---------- dependency install (cached unless pyproject.toml, uv.lock, or src/ changes) ----------
COPY cuopt_agent/pyproject.toml cuopt_agent/uv.lock /app/cuopt_agent/
COPY cuopt_agent/src /app/cuopt_agent/src

# Create venv that inherits system site-packages (cuOpt, numpy, numba, etc. from base image),
# then uv sync reads [tool.uv] overrides and pins to uv.lock — only installing what's missing.
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /app/cuopt_agent && \
    uv venv --system-site-packages && \
    uv sync --frozen

ENV PATH="/app/cuopt_agent/.venv/bin:$PATH"

# ---------- application code ----------
COPY cuopt_agent/configs /app/cuopt_agent/configs
COPY cuopt_agent/data    /app/cuopt_agent/data
COPY skills              /app/skills
COPY external/cuopt/skills /app/external/skills/cuopt
COPY AGENTS.md           /app/AGENTS.md

COPY cuopt_agent/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# nginx fronts NAT: routes POST /generate to /generate or /generate/stream
# based on the Accept header. NAT binds 127.0.0.1:8001; nginx listens on :8000.
# The njs body filter reshapes intermediate_data SSE events into data chunks.
COPY cuopt_agent/docker/nginx.conf       /etc/nginx/conf.d/cuopt-agent.conf
COPY cuopt_agent/docker/sse_transform.js /etc/nginx/njs/sse_transform.js
RUN rm -f /etc/nginx/sites-enabled/default /etc/nginx/conf.d/default.conf

RUN mkdir -p /config

# CWD must be cuopt_agent/ so config relative paths resolve correctly:
#   ../skills  -> /app/skills       ../AGENTS.md -> /app/AGENTS.md
WORKDIR /app/cuopt_agent

EXPOSE 8000

# DEBUG_LEVEL=1 (default) -> verbose logs (INFO + DEBUG); DEBUG_LEVEL=0 -> quiet (WARNING only).
ENV DEBUG_LEVEL=1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["nat", "serve", "--config_file", "/app/cuopt_agent/configs/config-deepagent.yml", "--host", "127.0.0.1", "--port", "8001"]
