#!/usr/bin/env bash
# Generate every run of one role from its recipe table:  generate.sh <development|holdout|crossroute|stress|ndt_covariance>
# For each recipe row: restart CARLA (required before every run), run run_scenario.sh, judge the run
# (probe OK, GO gate passed, >1000 logged rows), retry up to 4 times, and append to the role's ledger.
# Re-runnable: rows already marked OK in the ledger are skipped.
set +e
ROLE="$1"; [ -n "$ROLE" ] || { sed -n 2,5p "$0"; exit 3; }
DG="$(cd "$(dirname "$0")" && pwd)"; REL="$(dirname "$DG")"
[ -f "$REL/env.sh" ] && source "$REL/env.sh"
: "${CARLA_ROOT:?set CARLA_ROOT (see env.sh.example)}"; : "${ROS_SETUP:?set ROS_SETUP}"
RECIPES="$REL/data/indices/$ROLE.csv"; [ -f "$RECIPES" ] || { echo "no recipe table $RECIPES"; exit 3; }
BASE="${MELDLOC_DATA:-$REL/data}/logs/$ROLE"; LEDGER="$BASE/ledger.csv"; mkdir -p "$BASE"
[ -f "$LEDGER" ] || echo "scenario_id,replicate,attempt,probe,go_ok,csv_rows,run_dir,profile,covtype" > "$LEDGER"
case "$ROLE" in stress|ndt_covariance) export MAX_RUN_SEC=360 ;; *) export MAX_RUN_SEC=320 ;; esac

cleanup_ros() {
  ps -eo pid,args | grep -E "component_container|robot_state_publisher|map_hash_generator|autoware_[a-z_]+_node|lidar_bridge\.py|health_probe\.py|route_logger\.py|lidar_perturb_node\.py|ros2 launch" \
    | grep -vE "grep|generate\.sh" | awk -v me=$$ '$1 != me {print $1}' | xargs -r kill -9 2>/dev/null
  bash -c "source $ROS_SETUP; ros2 daemon stop" >/dev/null 2>&1
  sleep 4
}
start_carla() {
  ps -ef | grep "[C]arlaUE4" | awk '{print $2}' | xargs -r kill -9 2>/dev/null; sleep 4
  nohup bash -c "export DISPLAY=${DISPLAY:-:0} XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}; exec $CARLA_ROOT/CarlaUE4.sh -windowed -ResX=1280 -ResY=720 -quality-level=Low -nosound -carla-rpc-port=2000" > "$BASE/carla_server_last.log" 2>&1 &
  for i in $(seq 1 150); do
    python3 -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(3.0); print('rpc', c.get_server_version())" 2>/dev/null | grep -q rpc && return 0
    sleep 2
  done
  return 1
}

python3 "$DG/recipe_rows.py" "$RECIPES" | while IFS=";" read -r SID MODE SCHED SECT SEED ROUTE PROFILE COVTYPE REP; do
  [ -z "$SID" ] && continue
  NAME="${SID}_${PROFILE}_${COVTYPE}_r${REP}"
  grep -q "^${SID},${REP},.*,OK,1,.*,${PROFILE},${COVTYPE}$" "$LEDGER" && { echo "SKIP $NAME (done)"; continue; }
  ok=0
  for ATT in 1 2 3 4; do
    RUN="$BASE/${NAME}_a${ATT}"
    echo "RUN_START $NAME attempt=$ATT $(date +%H:%M:%S)"
    cleanup_ros
    start_carla || { echo "CARLA_START_FAIL $NAME a$ATT"; continue; }
    bash "$DG/run_scenario.sh" "$RUN" "$MODE" "$SCHED" "$SEED" "$SECT" "$ROUTE" "$PROFILE" "$COVTYPE" > "$RUN.console" 2>&1
    PR=$(grep -m1 -o "status=[A-Z_]*" "$RUN/post_init_probe2.txt" 2>/dev/null | cut -d= -f2)
    GO=$(grep -m1 -c "GO ok=True" "$RUN/long_route_pilot_summary.txt" 2>/dev/null)
    NR=$(wc -l < "$RUN/long_route_pilot.csv" 2>/dev/null || echo 0)
    echo "${SID},${REP},${ATT},${PR:-NONE},${GO:-0},${NR},${RUN},${PROFILE},${COVTYPE}" >> "$LEDGER"
    if [ "$PR" = "OK" ] && [ "${GO:-0}" -ge 1 ] && [ "${NR:-0}" -gt 1000 ]; then
      echo "RUN_DONE $NAME attempt=$ATT rows=$NR $(date +%H:%M:%S)"; ok=1; break
    fi
    echo "RUN_RETRY $NAME attempt=$ATT probe=${PR:-NONE} go=${GO:-0} rows=${NR}"
  done
  [ $ok -eq 0 ] && echo "RUN_FAIL $NAME (4 attempts)"
done
cleanup_ros
ps -ef | grep "[C]arlaUE4" | awk '{print $2}' | xargs -r kill -9 2>/dev/null
echo "BATCH_COMPLETE $ROLE"
