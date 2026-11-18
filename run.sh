#!/bin/sh
set -eu

# Resolve app.py and .venv relative to the launcher, even from another directory.
CDPATH='' cd -- "$(dirname -- "$0")"

if [ ! -x .venv/bin/python ]; then
    printf '%s\n' 'Set up .venv and install the dependencies as described in README.md.' >&2
    exit 1
fi

# Replace the shell with the Streamlit process and forward any extra arguments.
exec .venv/bin/python -m streamlit run app.py "$@"
