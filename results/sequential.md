| Phase | Subtask | Time taken |
|--------------------------|-------------------------------------|-----------------|
| Detection | Extract and pre-process the patch | 1.092 ms |
|  | Non-compressed custom descriptors | 0.212 ms |
|  | Compressed descriptors | 7.979 ms |
|  | Compressed custom descritors | 2.902 ms |
|  | Compress features and KRSL | 10.023 ms |
|  | Merge all features | 1.667 ms |
|  | **Compute the gaussian kernel** | **21.584** ms |
|  | Compute the FFT | 1.879 ms |
|  | Calculate filter response | 3.201 ms |
|  | Extract maximum response | 0.248 ms |
|  | *Total* | *50.787* ms |
| Extracting patches | Update bounding box | 0.000 ms |
|  | Non-compressed descriptors | 1.106 ms |
|  | Non-compressed custom descriptors | 0.196 ms |
|  | Compressed descriptors | 7.634 ms |
|  | Compressed custom descriptors | 3.113 ms |
|  | Update training data | 3.194 ms |
|  | *Total* | *15.131* ms |
| Feature compression | **Update projection matrix** | **21.409** ms |
|  | Compress | 4.413 ms |
|  | Merge all features | 0.714 ms |
|  | *Total* | *26.360* ms |
| Least Squares Regression | Initialization | 0.000 ms |
|  | **Calculate alphas** | **19.574** ms |
|  | Compute FFT | 1.864 ms |
|  | Add a small value | 0.391 ms |
|  | New Alphaf | 1.204 ms |
|  | Update RLS model | 0.998 ms |
|  | *Total* | *23.861* ms |
|  | ***Total time for a frame*** | ***117.713*** ms |
