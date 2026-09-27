### Worked example — pedestrians, frames 138 → 139, point (x, y) = (622, 119)

Window 7×7 centred on the point, full resolution, no pyramid. Rows are y (top to bottom), columns are x (left to right).

$I_1$ (frame 138 grey levels):

$$\begin{bmatrix}201 & 197 & 173 & 146 & 159 & 161 & 184 \\ 194 & 70 & 1 & 0 & 0 & 25 & 106 \\ 195 & 17 & 6 & 19 & 7 & 32 & 129 \\ 146 & 6 & 15 & 29 & 30 & 73 & 168 \\ 79 & 4 & 7 & 42 & 76 & 149 & 190 \\ 78 & 5 & 18 & 110 & 164 & 186 & 200 \\ 65 & 17 & 31 & 125 & 204 & 211 & 201\end{bmatrix}$$

$I_x = \frac{I_1(x+1,y) - I_1(x-1,y)}{2}$:

$$\begin{bmatrix}-5.5 & -14.0 & -25.5 & -7.0 & 7.5 & 12.5 & 21.0 \\ -68.5 & -96.5 & -35.0 & -0.5 & 12.5 & 53.0 & 79.5 \\ -91.0 & -94.5 & 1.0 & 0.5 & 6.5 & 61.0 & 80.0 \\ -88.5 & -65.5 & 11.5 & 7.5 & 22.0 & 69.0 & 61.0 \\ -63.5 & -36.0 & 19.0 & 34.5 & 53.5 & 57.0 & 18.0 \\ -58.0 & -30.0 & 52.5 & 73.0 & 38.0 & 18.0 & -9.0 \\ -33.5 & -17.0 & 54.0 & 86.5 & 43.0 & -1.5 & -25.5\end{bmatrix}$$

$I_y = \frac{I_1(x,y+1) - I_1(x,y-1)}{2}$:

$$\begin{bmatrix}-11.0 & -74.0 & -110.0 & -105.0 & -106.5 & -94.5 & -56.5 \\ -3.0 & -90.0 & -83.5 & -63.5 & -76.0 & -64.5 & -27.5 \\ -24.0 & -32.0 & 7.0 & 14.5 & 15.0 & 24.0 & 31.0 \\ -58.0 & -6.5 & 0.5 & 11.5 & 34.5 & 58.5 & 30.5 \\ -34.0 & -0.5 & 1.5 & 40.5 & 67.0 & 56.5 & 16.0 \\ -7.0 & 6.5 & 12.0 & 41.5 & 64.0 & 31.0 & 5.5 \\ -10.0 & 19.0 & 14.0 & -22.0 & -29.0 & -21.0 & -26.5\end{bmatrix}$$

$I_t = I_2(x,y) - I_1(x,y)$ at $d = 0$:

$$\begin{bmatrix}5 & 8 & -6 & -137 & -159 & -161 & -183 \\ 12 & 124 & 109 & 0 & 12 & 2 & -77 \\ 10 & 192 & 66 & -17 & -2 & 7 & -69 \\ 51 & 189 & 58 & -29 & -24 & 27 & -31 \\ 59 & 125 & 62 & -41 & -47 & 11 & 15 \\ 8 & 77 & 37 & -88 & -107 & -73 & -20 \\ -9 & 45 & 24 & -67 & -135 & -163 & -138\end{bmatrix}$$

Structure tensor (sums over the window):

$$G = \begin{bmatrix}\sum I_x^2 & \sum I_xI_y\\ \sum I_xI_y & \sum I_y^2\end{bmatrix} = \begin{bmatrix}116571.0 & 44162.5 \\ 44162.5 & 115601.5\end{bmatrix},\qquad \lambda_{min} = 71921.1,\ \lambda_{max} = 160251.4$$

Both eigenvalues are well above zero, so $G$ is invertible and the point is trackable (not an edge or a flat patch — the aperture problem does not apply here).

Iterations: $e = I_1(\mathbf{x}) - I_2(\mathbf{x} + \mathbf{d})$ (with $I_2$ sampled by bilinear interpolation), $\mathbf{b} = \sum [I_x e,\ I_y e]^T$, $\Delta\mathbf{d} = G^{-1}\mathbf{b}$, $\mathbf{d} \leftarrow \mathbf{d} + \Delta\mathbf{d}$.

| k | b | Δd | d after step | SSD before step |
|---|---|---|---|---|
| 1 | (99206.5, -20652.0) | (1.074, -0.589) | (1.074, -0.589) | 358758 |
| 2 | (49066.7, -45498.5) | (0.666, -0.648) | (1.741, -1.237) | 120807 |
| 3 | (-9545.7, -25304.7) | (0.001, -0.219) | (1.742, -1.457) | 22954 |
| 4 | (-265.3, -24.1) | (-0.003, 0.001) | (1.739, -1.456) | 18409 |

Final SSD over the window: 18446 (was 358758 at d = 0).

**Predicted** position in frame 139: (622.00, 119.00) + (1.739, -1.456) = **(623.74, 117.54)**.  

**Actual** position (patch correlation, independent of LK): **(623.79, 117.98)**. Difference: 0.44 px.
