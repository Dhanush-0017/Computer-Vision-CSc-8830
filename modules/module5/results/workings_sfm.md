## Part B — worked calculation (numbers from this run of `sfm.py`)

### Step 0 — camera parameters

Intrinsics from the Module 2 calibration (same phone, same main camera, full 3024×4032 resolution):

$$K = \begin{bmatrix}3023.59 & 0.00 & 1532.54 \\ 0.00 & 3028.34 & 1999.80 \\ 0.00 & 0.00 & 1.00\end{bmatrix}$$

Distortion $(k_1, k_2, p_1, p_2, k_3)$ = (0.1857, -0.9811, -0.0012, 0.0015, 1.4077). Every detected pixel is undistorted and converted to normalised coordinates $\mathbf{m} = K^{-1}\mathbf{x}$ first, so from here on the cameras are ideal pinholes with $K = I$.

Example, feature point 0 in view 1: undistorted pixel $(1275.94, 1312.62)$ → $\mathbf{m} = ((1275.94-1532.54)/3023.59,\ (1312.62-1999.80)/3028.34) = (-0.08487, -0.22692)$.

### Step 1 — homography view 1 → view 2 (DLT)

92 feature correspondences give a 184×9 matrix $A$. Its two smallest singular values are 6.37e+00 and 2.49e-02 — the smallest is far below the next, so the points are related by a single homography (they lie on a plane). $\mathbf{h}$ = the singular vector of the smallest:

$$H_{12} = \begin{bmatrix}-0.02882 & -0.59616 & -0.15075 \\ 0.60909 & -0.03885 & -0.06486 \\ -0.05073 & -0.10267 & 1.00000\end{bmatrix}$$

Singular values of $H_{12}$: 1.01892, 0.60710, 0.60168. Dividing by the middle one gives the Euclidean homography $\hat H = R + \mathbf{t}\mathbf{n}^T/d$:

$$\hat H_{12} = \begin{bmatrix}-0.04747 & -0.98199 & -0.24832 \\ 1.00328 & -0.06399 & -0.10684 \\ -0.08357 & -0.16911 & 1.64719\end{bmatrix}$$

### Step 2 — decomposition into $R, \mathbf{t}, \mathbf{n}$

From $\hat H^T\hat H$: $\sigma_1 = 1.67835$, $\sigma_2 = 1.00000$, $\sigma_3 = 0.99107$. The formulas in theory.md B4 give 4 solutions (checked against `cv2.decomposeHomographyMat`: largest difference 2.1e-15 over all three pairs).

$\hat H$ has 4 algebraic decompositions. Two put the plane behind camera 1 ($\mathbf{n}^T\mathbf{m} < 0$ for some point) and are rejected. The two left for each pair:

| pair | candidate | $\mathbf{n}$ (camera 1 frame) | $\mathbf{t}/d$ |
|---|---|---|---|
| 1→2 | 1 | (-0.215, -0.058, 0.975) | (-0.126, -0.037, 0.675) |
| 1→2 | 2 | (-0.040, 0.028, 0.999) | (-0.058, -0.153, 0.667) |
| 1→3 | 1 | (-0.151, -0.569, 0.809) | (-0.153, 0.305, 1.415) |
| 1→3 | 2 | (-0.122, 0.064, 0.990) | (-0.142, -0.647, 1.296) |
| 1→4 | 1 | (-0.318, -0.580, 0.750) | (-0.155, 0.407, 1.390) |
| 1→4 | 2 | (-0.095, 0.070, 0.993) | (-0.437, -0.607, 1.250) |

The plane is the same physical plane in all three pairs, so its normal in camera 1 must be the same, so the combination whose normals agree best is chosen: the three normals are 10.7° apart in total (sum of pairwise angles), against 20.6° for the next-best combination. Chosen:

$$R_{12} = \begin{bmatrix}-0.0498 & -0.9804 & -0.1908 \\ 0.9971 & -0.0597 & 0.0464 \\ -0.0569 & -0.1880 & 0.9805\end{bmatrix},\quad \mathbf{t}_{12}/d = \begin{bmatrix}-0.0576 \\ -0.1534 \\ 0.6675\end{bmatrix},\quad \mathbf{n} = \begin{bmatrix}-0.0400 \\ 0.0283 \\ 0.9988\end{bmatrix}$$

$$R_{13} = \begin{bmatrix}0.9990 & -0.0169 & 0.0405 \\ -0.0098 & 0.8133 & 0.5818 \\ -0.0428 & -0.5816 & 0.8123\end{bmatrix},\quad \mathbf{t}_{13}/d = \begin{bmatrix}-0.1422 \\ -0.6469 \\ 1.2964\end{bmatrix},\quad \mathbf{n} = \begin{bmatrix}-0.1220 \\ 0.0636 \\ 0.9905\end{bmatrix}$$

$$R_{14} = \begin{bmatrix}0.9860 & -0.0208 & 0.1656 \\ -0.0910 & 0.7649 & 0.6377 \\ -0.1399 & -0.6438 & 0.7523\end{bmatrix},\quad \mathbf{t}_{14}/d = \begin{bmatrix}-0.4370 \\ -0.6067 \\ 1.2500\end{bmatrix},\quad \mathbf{n} = \begin{bmatrix}-0.0950 \\ 0.0700 \\ 0.9930\end{bmatrix}$$

### Step 3 — triangulation of one boundary point

Panel corner TL. Normalised coordinates in the 4 views: (-0.1549, -0.3456), (0.0579, -0.1388), (-0.1055, -0.1459), (-0.1811, -0.0935). Camera matrices (after refinement) $P_1 = [I|\mathbf{0}]$, $P_i = [R_{1i}|\mathbf{t}_{1i}]$. Each view contributes $x\,\mathbf{p}_3^T - \mathbf{p}_1^T$ and $y\,\mathbf{p}_3^T - \mathbf{p}_2^T$:

$$A = \begin{bmatrix}-1.0000 & -0.0000 & -0.1549 & -0.0000 \\ -0.0000 & -1.0000 & -0.3456 & -0.0000 \\ 0.0546 & 0.9711 & 0.2394 & 0.1061 \\ -0.9958 & 0.0832 & -0.1441 & 0.0223 \\ -0.9965 & 0.0798 & -0.1081 & -0.0117 \\ 0.0030 & -0.7241 & -0.7049 & 0.4630 \\ -0.9587 & 0.1394 & -0.3069 & 0.2161 \\ 0.1086 & -0.6996 & -0.7124 & 0.4948\end{bmatrix}$$

Singular values of $A$: 2.04e+00, 2.00e+00, 7.72e-01, 3.85e-03. The last is $\approx 0$, as it should be. Its singular vector, divided by its 4th entry: $\mathbf{X} = (-0.1535, -0.3521, 1.0177)$ in units of $d$ (the distance from camera 1 to the plane).

The 92 feature points triangulated this way (from the homography cameras): RMS reprojection error **2.50 px**. After the refinement (re-estimate cameras 2–4 from the points, re-triangulate, 20 rounds): **0.81 px**. The 4 panel corners are then triangulated from the refined cameras: **4.19 px** RMS (they come from fitted edge lines, so they are located less precisely than the feature points).

### Step 4 — scale and the object frame

SfM gives shape only up to scale. One length fixes it: the top edge of the panel (TL to TR) is 170.0 mm (measured). In the reconstruction it is 0.38276 units long, so $s = 170.0 / 0.38276 = 444.14$ mm per unit ($d = 444$ mm).

Plane fitted to the points: $\mathbf{n} = (-0.1007, 0.0641, 0.9928)$. Object axes: $\mathbf{e}_1$ along the top edge (TL to TR), $\mathbf{e}_2 = \mathbf{n}\times\mathbf{e}_1$, origin at the TL corner. Object coordinates $= s\,[\mathbf{e}_1\ \mathbf{e}_2\ \mathbf{e}_3]^T(\mathbf{X}-\mathbf{X}_0)$.

Panel corners on the object (mm): TL (0.0, 0.0), TR (169.9, -0.0), BR (167.2, 232.8), BL (-2.6, 231.3). Out of plane: all within 4.93 mm.

Width = |TR − TL| = 169.9 mm (set by the scale), height = |BL − TL| = 231.3 mm against 230.0 mm measured; corner angles 90.7°, 89.3°, 90.2°, 89.9°.

### Step 5 — camera positions

Camera centre $\mathbf{C}_i = -R_{1i}^T\mathbf{t}_{1i}$, converted to the object frame and mm:

| view | image | C (mm) | distance to object (mm) | rotation vs view 1 |
|---|---|---|---|---|
| 1 | IMG_2005.jpg | (11, 186, -446) | 455 | 0.0° |
| 2 | IMG_2009.jpg | (33, 229, -747) | 755 | 93.7° |
| 3 | IMG_2012.jpg | (16, 772, -704) | 964 | 36.0° |
| 4 | IMG_2016.jpg | (201, 776, -640) | 926 | 41.8° |
