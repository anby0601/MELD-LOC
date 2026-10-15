# Data generation

Every simulation run is one row of a cohort table in `../data/indices/`. One shell
script per role regenerates that role's runs; each run restarts CARLA, drives the route with the
scheduled LiDAR perturbation, and logs the matcher's internal signals.

```
source ../env.sh                 # paths to ROS 2, the Autoware workspace, the maps and CARLA
./generate_development.sh        # 255 runs  (training + development-validation; split by the feature pipeline)
./generate_holdout.sh            #  24 runs  (in-route holdout, routeA, replicates r51/r52)
./generate_crossroute.sh         #  20 runs  (routeB / Town02, independently built map)
./generate_stress.sh             #  20 runs  (speed profiles + high-intensity recipes)
./generate_ndt_covariance.sh     #  15 runs  (holdout recipes with the matcher's Hessian covariance output enabled)
```

Runs are written to `$MELDLOC_DATA/logs/<role>/<scenario>_<profile>_<covtype>_r<rep>_a<attempt>/`
with a per-role `ledger.csv`. A run counts as valid when the second health probe reports `OK`,
the GO gate of the logger passes, and more than 1000 rows were logged; otherwise it is retried
up to four times. The scripts are re-runnable: valid rows in the ledger are skipped.

## What one run does (`run_scenario.sh`)

1. `nodes/lidar_bridge.py` connects to CARLA, loads the town, spawns the ego vehicle at the
   route start and drives the route with its speed profile while publishing the LiDAR scans.
2. `nodes/lidar_perturb_node.py` rewrites the scans according to the recipe
   (`mode`, `schedule` = route-progress-in-metres : intensity, `sector`, `seed`).
3. `launch/localization_only.launch.xml` starts the Autoware localization stack (NDT scan matcher,
   EKF, error monitor) on the perturbed cloud; `params/ndt_scan_matcher_<covtype>.param.yaml`
   selects the covariance option.
4. `nodes/direct_init.py` initializes the pose; `nodes/health_probe.py` checks the stack twice.
5. `nodes/route_logger.py` records the matcher outputs and the ground-truth pose at 10 Hz into
   `long_route_pilot.csv` until the route ends (or `MAX_RUN_SEC`: 320 s, 360 s for the stress and
   covariance sets, as in the original runs).

`commands.txt` in each run directory records the executed recipe; the recipe tables were built
from exactly these files for the archived runs.

## Reproduction notes

- CARLA is not deterministic across machines or restarts. Regenerated runs follow the same
  recipes but are not bit-identical to the archived logs (see the top-level README for the data
  archive).
- The archived cohorts were produced over several campaigns with earlier revisions of this
  runner. The perturbation node shipped here is the final one; the earlier revision used for the
  first 58 development runs differs only in lacking the non-rigid and longitudinal modes, and the
  rigid-mode kernels (`nodes/awsim_kernels/`) are identical. The logger shipped here is the
  final one (it adds the NDT covariance and execution-time columns used by the covariance analysis).
- `routeB` (Town02) requires the Town02 map built as described in `../env/`; its map is
  independent of the routeA map by construction.
- `tools/make_speed_profiles.py` derives `routes/routeA_fast.json` and `routes/routeA_aggr.json`
  from `routes/routeA.json`; the files are shipped, the script documents how.
