# X11 Display Forwarding for Python Development on macOS

## Summary

This note documents the setup for running Python code with graphical output (e.g., matplotlib, deep learning visualizations) inside Docker containers on macOS. The approach uses XQuartz as the X11 display server on the host and socat to create bidirectional communication streams between the Docker container and XQuartz.

## Key Points

- **XQuartz** provides X11 display capabilities on macOS and must be configured to allow network client connections.
- **socat** bridges the container's X11 display requests to the host's XQuartz UNIX socket, enabling GUI output to appear on the macOS display.
- The host's display IP is obtained from the `en0` interface, and the `$DISPLAY` variable is set accordingly.
- Helper scripts (`setDisplay.sh`, `socatStart.sh`, `startPythonDocker.sh`) automate the setup and container launch.

This complements the Python ML development environment setup by adding graphical rendering support for Docker-based workflows.

## Connections

Related to the [Python ML Development Environment](wiki/concepts/python-ml-development-environment.md) which covers the broader Python toolchain, and to distributed training concepts that may require visual debugging.