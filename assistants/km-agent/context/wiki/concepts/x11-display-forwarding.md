---
title: "X11 Display Forwarding for Python Development on macOS"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/coding/oldnotes.md]
related: [python-ml-development-environment]
tags: [docker, x11, xquartz, socat, macos, development-environment]
---

# X11 Display Forwarding for Python Development on macOS

Notes on configuring GUI support for Python code (e.g., matplotlib, graphics) running inside Docker containers on macOS, using XQuartz and socat.

## Setup Overview

The goal is to allow Python code executing in a Docker container to render graphical output on the host macOS display. This is achieved by:

1. **XQuartz**: Provides the X11 display server on macOS. Installed via Homebrew (`brew install xquartz`).
2. **socat**: Creates bidirectional streams between XQuartz and the Docker container endpoint. Installed via Homebrew (`brew install socat`).
3. **Docker container**: Runs the Python environment, with display forwarded to the host.

## Key Steps

- Install XQuartz and enable "allow connections from network clients" in X11 Preferences → Security tab.
- Determine the host's en0 interface IP address via `ifconfig en0 | grep "inet " | awk '{print $2}'`.
- Set the `$DISPLAY` environment variable to point to this IP.
- Use socat to bridge the Docker container's TCP port 6000 to the XQuartz UNIX socket: `socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\"`.
- Launch the Docker container with `./startPythonDocker.sh` and run Python scripts (e.g., `python deep-net-keras.py`).

## References
- Blog: [Running GUIs with Docker on Mac OS X](https://cntnr.io/running-guis-with-docker-on-mac-os-x-a14df6a76efc)

## Sources
- [Oldnotes](../summaries/oldnotes.md)

## Related
- [Python ML Development Environment](python-ml-development-environment.md)