#!/usr/bin/env bash
# One simulation run: CARLA bridge -> LiDAR perturbation node -> Autoware localization stack
# -> direct initialization -> two health probes -> route drive with logging. Blocks until the
# drive ends, then shuts the processes down. CARLA itself must already be running (see generate.sh).
#
#   run_scenario.sh RUN_DIR MODE SCHEDULE SEED SECTOR [ROUTE] [PROFILE] [COVTYPE]
#     MODE      perturbation-node mode (dropout, range_clip, sector_remove, sector_thin,
#               lateral_shear, yaw_shear, longitudinal_shear, translate_x, translate_y, yaw_rotate)
#     SCHEDULE  progress_m:value,...  (route progress in metres -> perturbation intensity)
#     SECTOR    sector direction for sector modes, "-" otherwise
#     ROUTE     routeA (Town01, default) | routeB (Town02)
#     PROFILE   std (default) | fast | aggr      (speed profile; routeA only)
#     COVTYPE   fixed (default) | laplace | laplace_raw | laplace_cal   (NDT covariance option)
#   env: ROS_SETUP AUTOWARE_WS AUTOWARE_MAP_DIR (from env.sh); MAX_RUN_SEC (default 320)
set +e
RUN="$1"; MODE="$2"; SCHED="$3"; SEED="$4"; SECTOR="$5"; ROUTE_ID="${6:-routeA}"; PROFILE="${7:-std}"; COVTYPE="${8:-fixed}"
[ -n "$RUN" ] && [ -n "$MODE" ] && [ -n "$SCHED" ] && [ -n "$SEED" ] && [ -n "$SECTOR" ] || { sed -n 2,16p "$0"; exit 3; }
DG="$(cd "$(dirname "$0")" && pwd)"; REL="$(dirname "$DG")"
[ -f "$REL/env.sh" ] && source "$REL/env.sh"
: "${ROS_SETUP:?set ROS_SETUP (see env.sh.example)}"; : "${AUTOWARE_WS:?set AUTOWARE_WS}"; : "${AUTOWARE_MAP_DIR:?set AUTOWARE_MAP_DIR}"
MAX_RUN_SEC="${MAX_RUN_SEC:-320}"
mkdir -p "$RUN"

case "$ROUTE_ID" in
  routeA) MAP=Town01; SPAWN="--x 383.18 --y -2.199 --yaw -180.0"
          case "$PROFILE" in
            std)  ROUTE=$DG/routes/routeA.json;      MAX_SPEED=6.0; CAP=5.5 ;;
            fast) ROUTE=$DG/routes/routeA_fast.json; MAX_SPEED=9.0; CAP=7.5 ;;
            aggr) ROUTE=$DG/routes/routeA_aggr.json; MAX_SPEED=6.0; CAP=5.5 ;;
            *) echo "bad PROFILE $PROFILE"; exit 3 ;;
          esac ;;
  routeB) MAP=Town02; SPAWN="--x -7.44 --y 247.24 --yaw 90.0"; ROUTE=$DG/routes/routeB.json; MAX_SPEED=6.0; CAP=5.5
          [ "$PROFILE" = "std" ] || { echo "routeB supports PROFILE=std only"; exit 3; } ;;
  *) echo "bad ROUTE $ROUTE_ID"; exit 3 ;;
esac
case "$COVTYPE" in
  fixed|laplace|laplace_raw|laplace_cal) NDT_PARAM=$DG/params/ndt_scan_matcher_${COVTYPE}.param.yaml ;;
  *) echo "bad COVTYPE $COVTYPE"; exit 3 ;;
esac
[ -f "$ROUTE" ] || { echo "ROUTE_MISSING $ROUTE"; exit 3; }
[ -f "$NDT_PARAM" ] || { echo "NDT_PARAM_MISSING $NDT_PARAM"; exit 3; }
[ -d "$AUTOWARE_MAP_DIR/$MAP" ] || { echo "MAP_MISSING $AUTOWARE_MAP_DIR/$MAP"; exit 3; }

SRC="source $ROS_SETUP; source $AUTOWARE_WS/install/setup.bash; export DISPLAY=${DISPLAY:-:0} XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)};"
RAW=/sensing/lidar/top/pointcloud_before_sync
DROP=/sensing/lidar/top/pointcloud_before_sync_dropout
SECT_ARG=""; [ "$SECTOR" != "-" ] && [ -n "$SECTOR" ] && SECT_ARG="--sector-direction $SECTOR"
echo "mode=$MODE schedule=$SCHED seed=$SEED sector=$SECTOR wall-sleep=0.05 max-speed=$MAX_SPEED profile=$PROFILE profile-cap=$CAP route=$ROUTE_ID covtype=$COVTYPE ndt-param=$(basename $NDT_PARAM) corridor=400 recorder=$MAX_RUN_SEC" > "$RUN/commands.txt"
grep -n "covariance_estimation_type" "$NDT_PARAM" >> "$RUN/commands.txt"

# 1) CARLA <-> ROS bridge (spawns the ego vehicle, publishes the LiDAR scans, drives the route)
nohup bash -c "$SRC cd $DG/nodes; exec python3 -u lidar_bridge.py --map $MAP $SPAWN --wall-sleep 0.05 --route-file '$ROUTE' --route-use-speed-profile --route-profile-speed-cap $CAP --max-speed $MAX_SPEED --route-corridor-distance 400 --spectator-follow" > "$RUN/bridge.log" 2>&1 &
BPID=$!
for i in $(seq 1 90); do grep -q "minimal bridge running" "$RUN/bridge.log" 2>/dev/null && { echo "bridge ready ${i}s"; break; }; kill -0 $BPID 2>/dev/null || { echo "BRIDGE_DIED"; exit 2; }; sleep 1; done
# 2) perturbation node (route-progress scheduled)
nohup bash -c "$SRC cd $DG/nodes; exec python3 -u lidar_perturb_node.py --mode $MODE --schedule '$SCHED' --seed $SEED $SECT_ARG --input-topic $RAW --output-topic $DROP --route-state-topic /carla_pilot/route_state --log-csv '$RUN/perturb_scan_log.csv'" > "$RUN/perturb_node.log" 2>&1 &
PPID2=$!; sleep 4
# 3) Autoware localization stack on the perturbed cloud
nohup bash -c "$SRC exec ros2 launch $DG/launch/localization_only.launch.xml map_path:=$AUTOWARE_MAP_DIR/$MAP input_pointcloud:=$DROP ndt_param_path:=$NDT_PARAM" > "$RUN/localization.log" 2>&1 &
LPID=$!
for i in $(seq 1 120); do
  kill -0 $LPID 2>/dev/null || { echo "LOC_DIED"; exit 2; }
  pm=$(grep -c 'Succeeded to load lanelet2_map' "$RUN/localization.log" 2>/dev/null)
  svc=$(bash -c "$SRC ros2 service list 2>/dev/null" | grep -c '/localization/initialize')
  [ "$svc" -ge 1 ] && [ "$pm" -ge 1 ] && { echo "loc ready ${i}s"; break; }
  sleep 2
done
for i in $(seq 1 60); do
  pcsvc=$(bash -c "$SRC ros2 service list 2>/dev/null" | grep -c 'pointcloud_map')
  [ "$pcsvc" -ge 1 ] && { echo "pointcloud map service ready ${i}s (n=$pcsvc)"; break; }
  sleep 1
done
sleep 10; echo "settle done"
# 4) direct initialization + two health probes (the second decides probe=OK in the ledger)
bash -c "$SRC cd $DG/nodes; python3 -u direct_init.py" 2>&1 | tee "$RUN/direct_init.log"
for k in 1 2; do
  OUT="$RUN/post_init_probe$([ $k = 2 ] && echo 2).txt"
  bash -c "$SRC cd $DG/nodes; python3 -u health_probe.py; echo probe_exit=\$?" 2>&1 | tee "$OUT"
done
# 5) route drive with logging (GO gate inside the logger; blocks until the route ends or MAX_RUN_SEC)
echo "=== recorder (GO gate + drive; blocks) ==="
bash -c "$SRC cd $DG/nodes; exec python3 -u route_logger.py --run-id $(basename $RUN) --csv '$RUN/long_route_pilot.csv' --summary '$RUN/long_route_pilot_summary.txt' --max-run-sec $MAX_RUN_SEC" > "$RUN/recorder.log" 2>&1
echo "recorder finished"
kill -INT $PPID2 $BPID 2>/dev/null; sleep 6
kill -INT $LPID 2>/dev/null; sleep 6
kill -TERM $PPID2 $BPID $LPID 2>/dev/null; sleep 2
pkill -TERM -f 'component_container_mt' 2>/dev/null; sleep 2
echo "DONE $RUN"
