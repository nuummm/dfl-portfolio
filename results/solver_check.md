# One-stage model vs. brute-force reference

Gurobi 13.0.3, 1 thread, time limit 60 s, T = 26, delta = 0.5. theta starts at the IPO solution. `ref` is the best of a 401 x 401 grid and the OLS / IPO points. `L(theta)` re-evaluates the returned theta with the exact lower-level solution. Runtimes are wall-clock on one laptop and only indicative.

| scenario | seed | formulation | theta | objective | L(theta) | ref | objective - ref | gap | time [s] | status |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| well-specified | 0 | kkt | (2.36, 2.31) | -0.638080 | -0.638080 | -0.638057 | -2.3e-05 | 0.0e+00 | 0.4 | optimal |
| well-specified | 0 | dual | (2.74, 2.66) | -0.637893 | -0.637893 | -0.638057 | +1.6e-04 | 1.8e-02 | 60.0 | maxTimeLimit |
| well-specified | 1 | kkt | (2.84, 3.29) | -0.506689 | -0.506689 | -0.506614 | -7.5e-05 | 0.0e+00 | 17.0 | optimal |
| well-specified | 1 | dual | (454, 558) | -0.506689 | -0.506689 | -0.506614 | -7.5e-05 | 3.3e-02 | 60.0 | maxTimeLimit |
| well-specified | 2 | kkt | (2.23, 2) | -0.649616 | -0.649616 | -0.649616 | +0.0e+00 | 0.0e+00 | 0.1 | optimal |
| well-specified | 2 | dual | (59.8, 49.7) | -0.649616 | -0.649616 | -0.649616 | -8.0e-10 | 8.9e-07 | 33.2 | optimal |
| well-specified | 3 | kkt | (8.41, 9.29) | -0.597292 | -0.597292 | -0.594671 | -2.6e-03 | 0.0e+00 | 12.6 | optimal |
| well-specified | 3 | dual | (2.33e+03, 2.57e+03) | -0.597292 | -0.597292 | -0.594671 | -2.6e-03 | 6.3e-02 | 60.0 | maxTimeLimit |
| well-specified | 4 | kkt | (2.61, 2.56) | -0.643384 | -0.643384 | -0.643376 | -7.4e-06 | 0.0e+00 | 0.6 | optimal |
| well-specified | 4 | dual | (39.5, 41.7) | -0.643167 | -0.643167 | -0.643376 | +2.1e-04 | 3.9e-02 | 60.0 | maxTimeLimit |
| nonlinear | 0 | kkt | (9.54, 8.92) | -0.680255 | -0.680255 | -0.661843 | -1.8e-02 | 0.0e+00 | 1.0 | optimal |
| nonlinear | 0 | dual | (20.2, 18.7) | -0.680255 | -0.680255 | -0.661843 | -1.8e-02 | 9.3e-02 | 2.5 | other |
| nonlinear | 1 | kkt | (11, 18.6) | -0.586971 | -0.586971 | -0.572589 | -1.4e-02 | 2.7e-07 | 0.8 | optimal |
| nonlinear | 1 | dual | (11, 18.6) | -0.586971 | -0.586971 | -0.572589 | -1.4e-02 | 6.6e-02 | 60.0 | maxTimeLimit |
| nonlinear | 2 | kkt | (1.95, 3.12) | -0.645242 | -0.645242 | -0.645178 | -6.4e-05 | 0.0e+00 | 0.1 | optimal |
| nonlinear | 2 | dual | (2.74, 4.37) | -0.645242 | -0.645242 | -0.645178 | -6.4e-05 | 1.2e-03 | 60.0 | maxTimeLimit |
| nonlinear | 3 | kkt | (3.26, 4.9) | -0.704846 | -0.704846 | -0.704724 | -1.2e-04 | 0.0e+00 | 0.9 | optimal |
| nonlinear | 3 | dual | (8.11, 11.9) | -0.704846 | -0.704846 | -0.704724 | -1.2e-04 | 4.9e-02 | 60.0 | maxTimeLimit |
| nonlinear | 4 | kkt | (3.24, 4.56) | -0.636145 | -0.636145 | -0.633028 | -3.1e-03 | 6.5e-07 | 0.1 | optimal |
| nonlinear | 4 | dual | (3.4, 4.78) | -0.636145 | -0.636145 | -0.633028 | -3.1e-03 | 1.0e-06 | 8.5 | optimal |
