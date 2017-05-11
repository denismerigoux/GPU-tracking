| Phase | Subtask | Time taken |
|--------------------------|-------------------------------------|-----------------|
| Detection | Extract and pre-process the patch | 1.021 ms |
|  | Non-compressed custom descriptors | 0.201 ms |
|  | Compressed descriptors | 7.758 ms |
|  | Compressed custom descritors | 2.841 ms |
|  | Compress features and KRSL | 9.998 ms |
|  | Merge all features | 1.569 ms |
|  | **Compute the gaussian kernel** | **20.714 ms** |
|  | Compute the FFT | 1.799 ms |
|  | Calculate filter response | 3.152 ms |
|  | Extract maximum response | 0.244 ms |
|  | *Total* | *49.297 ms* |
| Extracting patches | Update bounding box | 0.000 ms |
|  | Non-compressed descriptors | 1.027 ms |
|  | Non-compressed custom descriptors | 0.195 ms |
|  | Compressed descriptors | 7.433 ms |
|  | Compressed custom descriptors | 3.134 ms |
|  | Update training data | 3.097 ms |
|  | *Total* | *14.778 ms* |
| Feature compression | **Update projection matrix** | **20.615 ms** |
|  | Compress | 4.358 ms |
|  | Merge all features | 0.704 ms |
|  | *Total* | *25.504 ms* |
| Least Squares Regression | Initialization | 0.000 ms |
|  | **Calculate alphas** | **18.898 ms** |
|  | Compute FFT | 1.793 ms |
|  | Add a small value | 0.386 ms |
|  | New Alphaf | 1.143 ms |
|  | Update RLS model | 0.873 ms |
|  | *Total* | *22.924 ms* |
| ***Total time for a frame*** | | ***114.023 ms*** |
