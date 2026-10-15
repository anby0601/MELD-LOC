#!/usr/bin/env bash
# ndt_covariance: every run of data/indices/ndt_covariance.csv (see generate.sh)
exec bash "$(dirname "$0")/generate.sh" ndt_covariance
