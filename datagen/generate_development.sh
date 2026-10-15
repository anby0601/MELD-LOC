#!/usr/bin/env bash
# development: every run of data/indices/development.csv (see generate.sh)
exec bash "$(dirname "$0")/generate.sh" development
