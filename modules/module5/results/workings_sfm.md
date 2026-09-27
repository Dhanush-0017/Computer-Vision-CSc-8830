## Part B — worked calculation (numbers from this run of `sfm.py`)

### Step 0 — camera parameters

Intrinsics from the Module 2 calibration, scaled to the 1512×2016 images used here:

$$K = \begin{bmatrix}1511.79 & 0.00 & 766.02 \\ 0.00 & 1514.17 & 999.65 \\ 0.00 & 0.00 & 1.00\end{bmatrix}$$

Distortion $(k_1, k_2, p_1, p_2, k_3)$ = (0.1857, -0.9811, -0.0012, 0.0015, 1.4077). Every detected pixel is undistorted and converted to normalised coordinates $\mathbf{m} = K^{-1}\mathbf{x}$ first, so from here on the cameras are ideal pinholes with $K = I$.

Example, grid corner 0 in view 1: undistorted pixel $(574.36, 793.36)$ → $\mathbf{m} = ((574.36-766.02)/1511.79,\ (793.36-999.65)/1514.17) = (-0.12678, -0.13624)$.

### Step 1 — homography view 1 → view 2 (DLT)

54 grid correspondences give a 108×9 matrix $A$. Its two smallest singular values are 5.12e+00 and 1.43e-02 — one clearly near zero, so the points really are related by a single homography (they lie on a plane). $\mathbf{h}$ = the singular vector of the smallest:

$$H_{12} = \begin{bmatrix}1.35744 & -0.01325 & 0.16215 \\ 0.13468 & 0.83041 & 0.17140 \\ 0.82061 & -0.12093 & 1.00000\end{bmatrix}$$

Singular values of $H_{12}$: 1.75795, 0.84037, 0.70543. Dividing by the middle one gives the Euclidean homography $\hat H = R + \mathbf{t}\mathbf{n}^T/d$:

$$\hat H_{12} = \begin{bmatrix}1.61530 & -0.01577 & 0.19295 \\ 0.16026 & 0.98815 & 0.20395 \\ 0.97649 & -0.14391 & 1.18996\end{bmatrix}$$

### Step 2 — decomposition into $R, \mathbf{t}, \mathbf{n}$

From $\hat H^T\hat H$: $\sigma_1 = 2.09188$, $\sigma_2 = 1.00000$, $\sigma_3 = 0.83943$. The formulas in theory.md B4 give 4 solutions (checked against `cv2.decomposeHomographyMat`: largest difference 2.1e-15 over all three pairs).

$\hat H$ has 4 algebraic decompositions. Two put the plane behind camera 1 ($\mathbf{n}^T\mathbf{m} < 0$ for some point) and are rejected. The two left for each pair:

| pair | candidate | $\mathbf{n}$ (camera 1 frame) | $\mathbf{t}/d$ |
|---|---|---|---|
| 1→2 | 1 | (0.719, -0.025, 0.695) | (1.116, 0.131, 0.554) |
| 1→2 | 2 | (0.981, 0.028, 0.194) | (0.631, 0.142, 1.073) |
| 1→3 | 1 | (0.717, -0.034, 0.696) | (1.445, -0.048, 1.695) |
| 1→3 | 2 | (0.947, 0.085, 0.311) | (0.537, -0.232, 2.150) |
| 1→4 | 1 | (0.759, -0.036, 0.650) | (0.235, 0.216, 0.503) |
| 1→4 | 2 | (0.549, 0.296, 0.782) | (0.385, 0.026, 0.454) |

The plane is the same physical plane in all three pairs, so its normal in camera 1 must be the same. Only one combination agrees (sum of pairwise angles between the chosen normals: 7.55°); the other candidates disagree by tens of degrees. Chosen:

$$R_{12} = \begin{bmatrix}0.8131 & 0.0122 & -0.5820 \\ 0.0659 & 0.9914 & 0.1128 \\ 0.5784 & -0.1300 & 0.8054\end{bmatrix},\quad \mathbf{t}_{12}/d = \begin{bmatrix}1.1157 \\ 0.1313 \\ 0.5537\end{bmatrix},\quad \mathbf{n} = \begin{bmatrix}0.7190 \\ -0.0250 \\ 0.6946\end{bmatrix}$$

$$R_{13} = \begin{bmatrix}0.4704 & 0.1266 & -0.8733 \\ -0.2152 & 0.9762 & 0.0256 \\ 0.8558 & 0.1759 & 0.4865\end{bmatrix},\quad \mathbf{t}_{13}/d = \begin{bmatrix}1.4448 \\ -0.0476 \\ 1.6954\end{bmatrix},\quad \mathbf{n} = \begin{bmatrix}0.7173 \\ -0.0338 \\ 0.6960\end{bmatrix}$$

$$R_{14} = \begin{bmatrix}0.9962 & -0.0056 & -0.0874 \\ 0.0031 & 0.9996 & -0.0292 \\ 0.0875 & 0.0288 & 0.9957\end{bmatrix},\quad \mathbf{t}_{14}/d = \begin{bmatrix}0.2352 \\ 0.2163 \\ 0.5027\end{bmatrix},\quad \mathbf{n} = \begin{bmatrix}0.7590 \\ -0.0356 \\ 0.6501\end{bmatrix}$$

### Step 3 — triangulation of one boundary point

Screen corner TL. Normalised coordinates in the 4 views: (-0.2097, -0.2062), (-0.1430, -0.0334), (-0.1682, -0.1297), (-0.1471, -0.1059). Camera matrices (after refinement) $P_1 = [I|\mathbf{0}]$, $P_i = [R_{1i}|\mathbf{t}_{1i}]$. Each view contributes $x\,\mathbf{p}_3^T - \mathbf{p}_1^T$ and $y\,\mathbf{p}_3^T - \mathbf{p}_2^T$:

$$A = \begin{bmatrix}-1.0000 & -0.0000 & -0.2097 & -0.0000 \\ -0.0000 & -1.0000 & -0.2062 & -0.0000 \\ -0.8957 & -0.0020 & 0.4670 & -1.1952 \\ -0.0667 & -0.9909 & -0.1220 & -0.1755 \\ -0.5993 & -0.1517 & 0.8038 & -1.7507 \\ 0.1164 & -0.9989 & -0.0746 & -0.1929 \\ -1.0101 & 0.0002 & -0.0361 & -0.3225 \\ -0.0136 & -1.0039 & -0.0571 & -0.2801\end{bmatrix}$$

Singular values of $A$: 2.70e+00, 2.00e+00, 1.28e+00, 1.48e-03. The last is ≈ 0, as it should be. Its singular vector, divided by its 4th entry: $\mathbf{X} = (-0.3834, -0.3756, 1.8219)$ in units of $d$ (the distance from camera 1 to the plane).

The 54 grid points triangulated this way (from the homography cameras): RMS reprojection error **7.69 px**. After the refinement (re-estimate cameras 2–4 from the points, re-triangulate, 20 rounds): **0.12 px**. The 4 screen corners are then triangulated from the refined cameras: **1.99 px** RMS (their edges are soft glowing edges, so they are located less precisely than chessboard corners).

### Step 4 — scale and the object frame

SfM gives shape only up to scale. One length fixes it: grid corners 0 and 8 are 8 squares apart, 8 × 30.0 mm = 240.0 mm. In the reconstruction they are 0.31163 units apart, so $s = 240.0 / 0.31163 = 770.14$ mm per unit ($d = 770$ mm).

Plane fitted to the points: $\mathbf{n} = (0.7187, -0.0279, 0.6948)$. Object axes: $\mathbf{e}_1$ along the top row of the grid, $\mathbf{e}_2 = \mathbf{n}\times\mathbf{e}_1$, origin at grid corner 0. Object coordinates $= s\,[\mathbf{e}_1\ \mathbf{e}_2\ \mathbf{e}_3]^T(\mathbf{X}-\mathbf{X}_0)$.

Screen corners on the object (mm): TL (-202.1, -95.7), TR (441.8, -93.8), BR (440.6, 274.8), BL (-201.8, 275.8). Out of plane: all within 2.46 mm.

Width = |TR − TL| = 644.0 mm, height = |BL − TL| = 371.5 mm, aspect 1.738.

### Step 5 — camera positions

Camera centre $\mathbf{C}_i = -R_{1i}^T\mathbf{t}_{1i}$, converted to the object frame and mm:

| view | image | C (mm) | distance to object (mm) | rotation vs view 1 |
|---|---|---|---|---|
| 1 | IMG_1553.jpg | (1033, 42, -770) | 1195 | 0.0° |
| 2 | IMG_1554.jpg | (261, 17, -1350) | 1359 | 36.2° |
| 3 | IMG_1560.jpg | (-420, -201, -1696) | 1801 | 63.3° |
| 4 | IMG_1565.jpg | (1089, -169, -1165) | 1535 | 6.9° |
