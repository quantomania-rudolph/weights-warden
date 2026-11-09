#!/usr/bin/env bash
# ./h <command>  - front end of the weight-editing harness (see AGENTS.md). PYTHON=/path/to/python to choose the interpreter.
exec "${PYTHON:-python3}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/engine/harness_cli.py" "$@"
