# Cohort summary

| role | runs | expected | status |
|---|---|---|---|
| development | 255 | 255 | OK |
| holdout | 24 | 24 | OK |
| crossroute | 20 | 20 | OK |
| stress | 20 | 20 | OK |
| ndt_covariance | 15 | 15 | OK |

## Family counts

| role | family | runs |
|---|---|---|
| crossroute | lateral_shear | 2 |
| crossroute | random_dropout | 8 |
| crossroute | range_clip | 2 |
| crossroute | sector_remove | 4 |
| crossroute | yaw_rotate | 2 |
| crossroute | yaw_shear | 2 |
| development | lateral_shear | 30 |
| development | longitudinal_shear | 9 |
| development | random_dropout | 24 |
| development | range_clip | 47 |
| development | sector_remove | 40 |
| development | sector_thin | 24 |
| development | translate_x | 2 |
| development | translate_y | 22 |
| development | yaw_rotate | 27 |
| development | yaw_shear | 30 |
| holdout | lateral_shear | 2 |
| holdout | random_dropout | 10 |
| holdout | range_clip | 2 |
| holdout | sector_remove | 6 |
| holdout | yaw_rotate | 2 |
| holdout | yaw_shear | 2 |
| ndt_covariance | healthy | 2 |
| ndt_covariance | lateral_shear | 1 |
| ndt_covariance | random_dropout | 6 |
| ndt_covariance | range_clip | 1 |
| ndt_covariance | sector_remove | 3 |
| ndt_covariance | yaw_rotate | 1 |
| ndt_covariance | yaw_shear | 1 |
| stress | healthy | 6 |
| stress | lateral_shear | 3 |
| stress | random_dropout | 3 |
| stress | range_clip | 3 |
| stress | sector_remove | 1 |
| stress | yaw_rotate | 1 |
| stress | yaw_shear | 3 |

## Checks

- development runs not in the frozen feature cache (frozen_cache = 0): 18
- run directories without commands.txt: 0
- index/commands.txt recipe mismatches: 0
