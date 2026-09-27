### Worked example — store_aisle, frames 800 → 801, point (x, y) = (516, 124)

Window 7×7 centred on the point, full resolution, no pyramid. Rows are y (top to bottom), columns are x (left to right).

$I_1$ (frame 800 grey levels):

$$\begin{bmatrix}94 & 105 & 119 & 116 & 103 & 84 & 68 \\ 50 & 66 & 93 & 94 & 80 & 68 & 65 \\ 38 & 40 & 49 & 46 & 45 & 51 & 59 \\ 50 & 38 & 18 & 12 & 20 & 31 & 55 \\ 33 & 29 & 17 & 18 & 19 & 26 & 66 \\ 19 & 20 & 20 & 22 & 22 & 42 & 103 \\ 15 & 19 & 22 & 18 & 22 & 63 & 138\end{bmatrix}$$

$I_x = \frac{I_1(x+1,y) - I_1(x-1,y)}{2}$:

$$\begin{bmatrix}-4.0 & 12.5 & 5.5 & -8.0 & -16.0 & -17.5 & -11.0 \\ -6.0 & 21.5 & 14.0 & -6.5 & -13.0 & -7.5 & -3.0 \\ -6.0 & 5.5 & 3.0 & -2.0 & 2.5 & 7.0 & 6.0 \\ -6.5 & -16.0 & -13.0 & 1.0 & 9.5 & 17.5 & 24.0 \\ -1.0 & -8.0 & -5.5 & 1.0 & 4.0 & 23.5 & 39.0 \\ 2.0 & 0.5 & 1.0 & 1.0 & 10.0 & 40.5 & 49.5 \\ 2.5 & 3.5 & -0.5 & 0.0 & 22.5 & 58.0 & 52.0\end{bmatrix}$$

$I_y = \frac{I_1(x,y+1) - I_1(x,y-1)}{2}$:

$$\begin{bmatrix}-38.0 & -33.5 & -20.5 & -19.0 & -21.5 & -12.0 & -10.5 \\ -28.0 & -32.5 & -35.0 & -35.0 & -29.0 & -16.5 & -4.5 \\ 0.0 & -14.0 & -37.5 & -41.0 & -30.0 & -18.5 & -5.0 \\ -2.5 & -5.5 & -16.0 & -14.0 & -13.0 & -12.5 & 3.5 \\ -15.5 & -9.0 & 1.0 & 5.0 & 1.0 & 5.5 & 24.0 \\ -9.0 & -5.0 & 2.5 & 0.0 & 1.5 & 18.5 & 36.0 \\ -2.5 & -1.0 & 3.0 & -6.0 & 1.0 & 37.5 & 33.0\end{bmatrix}$$

$I_t = I_2(x,y) - I_1(x,y)$ at $d = 0$:

$$\begin{bmatrix}-22 & -18 & -40 & -49 & -39 & -20 & -10 \\ -11 & -29 & -58 & -52 & -28 & -5 & 5 \\ -11 & -28 & -35 & -24 & -4 & 21 & 36 \\ -28 & -19 & 3 & 9 & 29 & 68 & 67 \\ -14 & -7 & 5 & 9 & 57 & 113 & 81 \\ 3 & -2 & -7 & 20 & 92 & 127 & 59 \\ 10 & -2 & -16 & 56 & 133 & 118 & 36\end{bmatrix}$$

Structure tensor (sums over the window):

$$G = \begin{bmatrix}\sum I_x^2 & \sum I_xI_y\\ \sum I_xI_y & \sum I_y^2\end{bmatrix} = \begin{bmatrix}16507.0 & 7447.8 \\ 7447.8 & 20120.0\end{bmatrix},\qquad \lambda_{min} = 10649.8,\ \lambda_{max} = 25977.2$$

Both eigenvalues are well above zero, so $G$ is invertible and the point is trackable (not an edge or a flat patch — the aperture problem does not apply here).

Iterations: $e = I_1(\mathbf{x}) - I_2(\mathbf{x} + \mathbf{d})$ (with $I_2$ sampled by bilinear interpolation), $\mathbf{b} = \sum [I_x e,\ I_y e]^T$, $\Delta\mathbf{d} = G^{-1}\mathbf{b}$, $\mathbf{d} \leftarrow \mathbf{d} + \Delta\mathbf{d}$.

| k | b | Δd | d after step | SSD before step |
|---|---|---|---|---|
| 1 | (-31042.0, -24462.5) | (-1.599, -0.624) | (-1.599, -0.624) | 118213 |
| 2 | (-3654.4, -10103.7) | (0.006, -0.504) | (-1.593, -1.128) | 6842 |
| 3 | (459.6, -973.7) | (0.060, -0.070) | (-1.533, -1.199) | 962 |
| 4 | (-270.5, -154.2) | (-0.016, -0.002) | (-1.549, -1.201) | 871 |
| 5 | (32.3, 28.5) | (0.002, 0.001) | (-1.547, -1.200) | 884 |

Final SSD over the window: 881 (was 118213 at d = 0).

**Predicted** position in frame 801: (516.00, 124.00) + (-1.547, -1.200) = **(514.45, 122.80)**.  

**Actual** position (patch correlation, independent of LK): **(514.55, 122.61)**. Difference: 0.22 px.
