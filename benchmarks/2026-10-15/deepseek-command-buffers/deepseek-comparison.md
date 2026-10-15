| Mode          | Phase   | Mean (ms) | Throughput (tokens/s) | Speedup vs OFF | Performance change | CV (%) | Drift (%) | Status   |
| ------------- | ------- | --------- | --------------------- | -------------- | ------------------ | ------ | --------- | -------- |
| OFF           | prefill | 836.633   | N/A                   | 1.000×         | +0.00%             | 0.368  | 0.175     | stable   |
| OFF           | decode  | 95.902    | 41.709                | 1.000×         | +0.00%             | 6.997  | 2.089     | stable   |
| ALWAYS_UPDATE | prefill | 974.854   | N/A                   | 0.858×         | -14.18%            | 1.336  | 0.004     | stable   |
| ALWAYS_UPDATE | decode  | N/A       | N/A                   | N/A            | N/A                | 6.030  | 10.676    | unstable |
| SKIP_TEMP     | prefill | 927.650   | N/A                   | 0.902×         | -9.81%             | 1.167  | 0.796     | stable   |
| SKIP_TEMP     | decode  | 119.762   | 33.400                | 0.801×         | -19.92%            | 1.476  | 0.690     | stable   |
| SKIP_PROFILED | prefill | 963.774   | N/A                   | 0.868×         | -13.19%            | 0.809  | 0.159     | stable   |
| SKIP_PROFILED | decode  | 99.681    | 40.128                | 0.962×         | -3.79%             | 0.613  | 0.024     | stable   |
