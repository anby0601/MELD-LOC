#!/usr/bin/env bash
# stress: every run of data/indices/stress.csv (see generate.sh)
exec bash "$(dirname "$0")/generate.sh" stress
