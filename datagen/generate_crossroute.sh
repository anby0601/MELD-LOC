#!/usr/bin/env bash
# crossroute: every run of data/indices/crossroute.csv (see generate.sh)
exec bash "$(dirname "$0")/generate.sh" crossroute
