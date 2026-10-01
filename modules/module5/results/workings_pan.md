### Worked example — pan, frames 308 → 309, point (x, y) = (259, 198)

The point is on the static background, and moves only because the camera turns.

Window 7×7 centred on the point, full resolution, no pyramid. Rows are y (top to bottom), columns are x (left to right).

$I_1$ (frame 308 grey levels):

$$\begin{bmatrix}113 & 69 & 179 & 241 & 211 & 241 & 98 \\ 44 & 12 & 87 & 231 & 238 & 234 & 118 \\ 67 & 16 & 103 & 242 & 233 & 248 & 131 \\ 159 & 126 & 206 & 234 & 159 & 217 & 194 \\ 221 & 239 & 225 & 140 & 54 & 125 & 217 \\ 67 & 78 & 64 & 59 & 79 & 84 & 71 \\ 69 & 100 & 104 & 123 & 149 & 154 & 126\end{bmatrix}$$

$I_x = \frac{I_1(x+1,y) - I_1(x-1,y)}{2}$:

$$\begin{bmatrix}-81.0 & 33.0 & 86.0 & 16.0 & 0.0 & -56.5 & -108.5 \\ -84.5 & 21.5 & 109.5 & 75.5 & 1.5 & -60.0 & -102.5 \\ -96.0 & 18.0 & 113.0 & 65.0 & 3.0 & -51.0 & -115.0 \\ -62.5 & 23.5 & 54.0 & -23.5 & -8.5 & 17.5 & -47.0 \\ 0.0 & 2.0 & -49.5 & -85.5 & -7.5 & 81.5 & 52.0 \\ -48.0 & -1.5 & -9.5 & 7.5 & 12.5 & -4.0 & -5.0 \\ 1.5 & 17.5 & 11.5 & 22.5 & 15.5 & -11.5 & -27.0\end{bmatrix}$$

$I_y = \frac{I_1(x,y+1) - I_1(x,y-1)}{2}$:

$$\begin{bmatrix}-91.0 & -106.5 & -68.0 & 32.0 & 28.0 & 17.0 & 10.5 \\ -23.0 & -26.5 & -38.0 & 0.5 & 11.0 & 3.5 & 16.5 \\ 57.5 & 57.0 & 59.5 & 1.5 & -39.5 & -8.5 & 38.0 \\ 77.0 & 111.5 & 61.0 & -51.0 & -89.5 & -61.5 & 43.0 \\ -46.0 & -24.0 & -71.0 & -87.5 & -40.0 & -66.5 & -61.5 \\ -76.0 & -69.5 & -60.5 & -8.5 & 47.5 & 14.5 & -45.5 \\ 44.0 & 46.5 & 48.5 & 56.5 & 47.0 & 45.5 & 51.0\end{bmatrix}$$

$I_t = I_2(x,y) - I_1(x,y)$ at $d = 0$:

$$\begin{bmatrix}128 & 63 & -105 & -94 & 35 & -49 & 141 \\ 164 & 45 & -86 & -174 & -18 & 11 & 120 \\ 162 & 71 & -87 & -172 & -7 & -4 & 116 \\ 95 & 68 & -57 & -31 & 92 & -59 & 10 \\ 26 & -22 & 3 & 79 & 98 & -64 & -128 \\ 132 & 11 & 15 & 8 & -24 & -12 & 5 \\ 43 & -39 & -20 & -22 & -35 & -5 & 35\end{bmatrix}$$

Structure tensor (sums over the window):

$$G = \begin{bmatrix}\sum I_x^2 & \sum I_xI_y\\ \sum I_xI_y & \sum I_y^2\end{bmatrix} = \begin{bmatrix}146416.8 & -784.2 \\ -784.2 & 141626.8\end{bmatrix},\qquad \lambda_{min} = 141501.6,\ \lambda_{max} = 146541.9$$

Both eigenvalues are well above zero, so $G$ is invertible and the point is trackable (not an edge or a flat patch — the aperture problem does not apply here).

Iterations: $e = I_1(\mathbf{x}) - I_2(\mathbf{x} + \mathbf{d})$ (with $I_2$ sampled by bilinear interpolation), $\mathbf{b} = \sum [I_x e,\ I_y e]^T$, $\Delta\mathbf{d} = G^{-1}\mathbf{b}$, $\mathbf{d} \leftarrow \mathbf{d} + \Delta\mathbf{d}$.

| k | b | Δd | d after step | SSD before step |
|---|---|---|---|---|
| 1 | (166239.0, 6210.5) | (1.136, 0.050) | (1.136, 0.050) | 319824 |
| 2 | (17405.0, -13601.1) | (0.118, -0.095) | (1.254, -0.045) | 9901 |
| 3 | (-1414.7, 1043.8) | (-0.010, 0.007) | (1.244, -0.038) | 8584 |
| 4 | (102.6, -125.7) | (0.001, -0.001) | (1.245, -0.039) | 8228 |

Final SSD over the window: 8257 (was 319824 at d = 0).

**Predicted** position in frame 309: (259.00, 198.00) + (1.245, -0.039) = **(260.25, 197.96)**.  

**Actual** position (patch correlation, independent of LK): **(260.13, 198.00)**. Difference: 0.12 px.
