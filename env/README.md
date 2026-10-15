# Simulation environment

Every archived run (the 299 cohort runs and the 35 evaluation-only runs) was executed with
CARLA 0.9.15 and the Autoware workspace pinned below (ROS 2 Humble, Ubuntu 22.04, built with
`colcon build --symlink-install`). This folder is the record needed to rebuild that workspace.

## Sub-repository pins

| path | origin | commit | local mods | untracked adds |
|---|---|---|---|---|
| `src/core/autoware_adapi_msgs` | https://github.com/autowarefoundation/autoware_adapi_msgs.git | `6c3faf2311db` | 0 | 0 |
| `src/core/autoware_cmake` | https://github.com/autowarefoundation/autoware_cmake.git | `ac910018c8b7` | 0 | 0 |
| `src/core/autoware_core` | https://github.com/autowarefoundation/autoware_core.git | `f9d099eeb569` | 1 | 0 |
| `src/core/autoware_internal_msgs` | https://github.com/autowarefoundation/autoware_internal_msgs.git | `094122d37e56` | 0 | 0 |
| `src/core/autoware_lanelet2_extension` | https://github.com/autowarefoundation/autoware_lanelet2_extension.git | `38b5af5ac812` | 0 | 0 |
| `src/core/autoware_msgs` | https://github.com/autowarefoundation/autoware_msgs.git | `588f00df3acc` | 0 | 0 |
| `src/core/autoware_rviz_plugins` | https://github.com/autowarefoundation/autoware_rviz_plugins.git | `5d3d4f0aa296` | 0 | 0 |
| `src/core/autoware_system_designer` | https://github.com/autowarefoundation/autoware_system_designer.git | `5d5faa656c83` | 0 | 0 |
| `src/core/autoware_utils` | https://github.com/autowarefoundation/autoware_utils.git | `2c663f0274f5` | 0 | 0 |
| `src/core/external/rviz_2d_overlay_plugins` | https://github.com/teamspatzenhirn/rviz_2d_overlay_plugins.git | `36b3c86d37af` | 0 | 0 |
| `src/launcher/autoware_launch` | https://github.com/autowarefoundation/autoware_launch.git | `10d1251ecea0` | 15 | 3 |
| `src/launcher/autoware_launch/sensor_kit/carla_sensor_kit_launch` | https://github.com/mraditya01/carla_sensor_kit_launch.git | `4e1186b2800a` | 0 | 0 |
| `src/middleware/external/agnocast` | https://github.com/tier4/agnocast.git | `27ad575790ee` | 0 | 0 |
| `src/sensor_component/external/nebula` | https://github.com/tier4/nebula.git | `9cb61cc96581` | 0 | 0 |
| `src/sensor_component/external/sensor_component_description` | https://github.com/tier4/sensor_component_description.git | `03ba094851ec` | 0 | 0 |
| `src/sensor_component/external/sync_tooling_msgs` | https://github.com/tier4/sync_tooling_msgs.git | `a1d519062e76` | 0 | 0 |
| `src/sensor_component/ros2_socketcan` | https://github.com/autowarefoundation/ros2_socketcan | `e39a814180b0` | 0 | 0 |
| `src/sensor_component/transport_drivers` | https://github.com/autowarefoundation/transport_drivers | `39ebd8afe1bb` | 0 | 0 |
| `src/sensor_kit/sample_sensor_kit_launch` | https://github.com/autowarefoundation/sample_sensor_kit_launch.git | `b131cc4de513` | 10 | 0 |
| `src/universe/autoware_universe` | https://github.com/autowarefoundation/autoware_universe.git | `14ae8fd36cf5` | 2 | 0 |
| `src/universe/external/bevdet_vendor` | https://github.com/autowarefoundation/bevdet_vendor.git | `143465b355ef` | 1 | 0 |
| `src/universe/external/cuda_blackboard` | https://github.com/autowarefoundation/cuda_blackboard.git | `e3f13d2ddb0c` | 0 | 0 |
| `src/universe/external/eagleye` | https://github.com/MapIV/eagleye.git | `575136ebba99` | 0 | 0 |
| `src/universe/external/glog` | https://github.com/tier4/glog.git | `ea36766fdc2a` | 0 | 0 |
| `src/universe/external/llh_converter` | https://github.com/MapIV/llh_converter.git | `4fc2a2e1bc9d` | 0 | 0 |
| `src/universe/external/managed_transform_buffer` | https://github.com/autowarefoundation/managed_transform_buffer.git | `b12c9c11bccd` | 0 | 0 |
| `src/universe/external/morai_msgs` | https://github.com/MORAI-Autonomous/MORAI-ROS2_morai_msgs.git | `e2e75fc1603a` | 0 | 0 |
| `src/universe/external/muSSP` | https://github.com/tier4/muSSP.git | `c79e98fd5e65` | 0 | 0 |
| `src/universe/external/negotiated` | https://github.com/osrf/negotiated.git | `eac198b55dcd` | 0 | 0 |
| `src/universe/external/pointcloud_to_laserscan` | https://github.com/tier4/pointcloud_to_laserscan.git | `d969ec699f84` | 0 | 0 |
| `src/universe/external/rtklib_ros_bridge` | https://github.com/MapIV/rtklib_ros_bridge.git | `ef094407bba4` | 0 | 0 |
| `src/universe/external/tier4_autoware_msgs` | https://github.com/tier4/tier4_autoware_msgs.git | `b9c385197baf` | 0 | 0 |
| `src/universe/external/trt_batched_nms` | https://github.com/autowarefoundation/trt_batched_nms.git | `98024a240c63` | 0 | 0 |

## Local modifications

Repos with local mods carry parameter/launch tuning for the CARLA setup (NDT scan matcher,
EKF, crop box, pose initializer) plus two build fixes. Exact diffs: `patches/*.patch`
(regenerated 2026-09-11 from the live workspace; identical to the 2026-05-30 snapshot).

Untracked additions inside `src/launcher/autoware_launch` (the CARLA sensor kit package
`carla_sensor_kit_launch`, the `tier4_universe_launch` launcher subtree, and one diagnostic
default param) are archived verbatim in `autoware_launch_untracked_additions.tar.gz`.

## Reproduction recipe

1. Clone each repo at the pinned commit into the listed path.
2. `git apply` the matching patch in each modified repo.
3. Extract `autoware_launch_untracked_additions.tar.gz` into `src/launcher/autoware_launch/`.
4. Copy `individual_params/` (in this folder) to the workspace root.
5. `colcon build --symlink-install`; point `AUTOWARE_WS` in `../env.sh` at the workspace.

## CARLA

CARLA 0.9.15 binary release; `CarlaUE4.sh` is started by `datagen/generate.sh` with
`-quality-level=Low -nosound -carla-rpc-port=2000`. The `carla` Python egg of that release must be
importable by the system `python3` used for the generation scripts (`import carla` is the readiness probe).

## Maps

The localization stack loads `$AUTOWARE_MAP_DIR/<Town>/{pointcloud_map.pcd, lanelet2_map.osm, map_projector_info.yaml}`.
The maps are part of the data archive. They were built with the two tools in `../datagen/tools/`:
`mapping_sweep.py` accumulates LiDAR scans at ground-truth poses along every lane (the point clouds
shipped with CARLA do not register well enough with the run-time LiDAR), and `xodr_to_lanelet2.py`
converts the town's OpenDRIVE file into the lanelet2 map. Town01 (routeA, 16.7 M points) and Town02
(routeB, 8.4 M points, 1 m waypoint spacing, 0.05 m voxel) were built independently.
