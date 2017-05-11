| Phase | Subtask | Time taken |
|--------------------------|-------------------------------------|-----------------|
| Detection | Extract and pre-process the patch | 1.028 ms |
|  | Non-compressed custom descriptors | 0.204 ms |
|  | Compressed descriptors | 7.796 ms |
|  | Compressed custom descritors | 2.821 ms |
|  | Compress features and KRSL | 9.986 ms |
|  | Merge all features | 1.595 ms |
|  | **Compute the gaussian kernel** | **21.149 ms** |
|  | Compute the FFT | 1.874 ms |
|  | Calculate filter response | 3.171 ms |
|  | Extract maximum response | 0.248 ms |
|  | *Total* | *49.873 ms* |
| Extracting patches | Update bounding box | 0.000 ms |
|  | Non-compressed descriptors | 1.023 ms |
|  | Non-compressed custom descriptors | 0.199 ms |
|  | Compressed descriptors | 7.397 ms |
|  | Compressed custom descriptors | 3.009 ms |
|  | Update training data | 3.153 ms |
|  | *Total* | *14.666 ms* |
| Feature compression | **Update projection matrix** | **20.677 ms** |
|  | Compress | 4.404 ms |
|  | Merge all features | 0.717 ms |
|  | *Total* | *25.623 ms* |
| Least Squares Regression | Initialization | 0.000 ms |
|  | **Calculate alphas** | **19.000 ms** |
|  | Compute FFT | 1.847 ms |
|  | Add a small value | 0.380 ms |
|  | New Alphaf | 1.135 ms |
|  | Update RLS model | 0.923 ms |
|  | *Total* | *23.115 ms* |
|  | ***Total time for a frame*** | ***114.811 ms*** |
