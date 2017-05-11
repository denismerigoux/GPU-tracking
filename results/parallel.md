| Phase | Subtask | Time taken |
|--------------------------|-------------------------------------|-----------------|
| Detection | Extract and pre-process the patch | 1.049 ms |
|  | Non-compressed custom descriptors | 0.207 ms |
|  | **Compressed descriptors** | **3.129 ms** |
|  | Compressed custom descritors | 2.887 ms |
|  | Compress features and KRSL | 9.741 ms |
|  | Merge all features | 1.618 ms |
|  | **Compute the gaussian kernel** | **11.072 ms** |
|  | Compute the FFT | 1.934 ms |
|  | Calculate filter response | 3.248 ms |
|  | Extract maximum response | 0.252 ms |
|  | *Total* | *35.135 ms* |
| Extracting patches | Update bounding box | 0.000 ms |
|  | Non-compressed descriptors | 1.001 ms |
|  | Non-compressed custom descriptors | 0.198 ms |
|  | **Compressed descriptors** | **3.125 ms** |
|  | Compressed custom descriptors | 3.578 ms |
|  | Update training data | 3.102 ms |
|  | *Total* | *10.909 ms* |
| Feature compression | **Update projection matrix** | **14.010 ms** |
|  | Compress | 4.465 ms |
|  | Merge all features | 0.719 ms |
|  | *Total* | *17.932 ms* |
| Least Squares Regression | Initialization | 0.000 ms |
|  | **Calculate alphas** | **12.068 ms** |
|  | Compute FFT | 1.924 ms |
|  | Add a small value | 0.400 ms |
|  | New Alphaf | 1.168 ms |
|  | Update RLS model | 0.864 ms |
|  | *Total* | *15.034 ms* |
| Total time for a frame | ****** | ***80.548 ms*** |
