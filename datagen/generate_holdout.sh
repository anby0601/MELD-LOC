#!/usr/bin/env bash
# holdout: every run of data/indices/holdout.csv (see generate.sh)
exec bash "$(dirname "$0")/generate.sh" holdout
