| Phase | Subtask | Time taken |
|--------------------------|-------------------------------------|-----------------|
| Detection | Extract and pre-process the patch | 1.135 ms |
|  | Non-compressed custom descriptors | 0.204 ms |
|  | Compressed descriptors | 6.620 ms |
|  | Compressed custom descritors | 2.826 ms |
|  | Compress features and KRSL | 9.216 ms |
|  | Merge all features | 1.593 ms |
|  | **Compute the gaussian kernel** | **10.899** ms |
|  | Compute the FFT | 1.864 ms |
|  | Calculate filter response | 3.155 ms |
|  | Extract maximum response | 0.240 ms |
|  | *Total* | *37.751* ms |
| Extracting patches | Update bounding box | 0.000 ms |
|  | Non-compressed descriptors | 1.013 ms |
|  | Non-compressed custom descriptors | 0.196 ms |
|  | Compressed descriptors | 6.392 ms |
|  | Compressed custom descriptors | 3.468 ms |
|  | Update training data | 3.047 ms |
|  | *Total* | *14.017* ms |
| Feature compression | **Update projection matrix** | **14.028** ms |
|  | Compress | 4.345 ms |
|  | Merge all features | 0.717 ms |
|  | *Total* | *17.827* ms |
| Least Squares Regression | Initialization | 0.000 ms |
|  | **Calculate alphas** | **11.926** ms |
|  | Compute FFT | 1.872 ms |
|  | Add a small value | 0.412 ms |
|  | New Alphaf | 1.167 ms |
|  | Update RLS model | 0.922 ms |
|  | *Total* | *14.881* ms |
|  | ***Total time for a frame*** | ***86.017*** ms |
