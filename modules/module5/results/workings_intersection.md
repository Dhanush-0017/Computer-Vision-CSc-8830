### Worked example — intersection, frames 874 → 875, point (x, y) = (207, 324)

The point is on a moving car.

Window 7×7 centred on the point, full resolution, no pyramid. Rows are y (top to bottom), columns are x (left to right).

$I_1$ (frame 874 grey levels):

$$\begin{bmatrix}133 & 152 & 229 & 255 & 252 & 255 & 255 \\ 117 & 130 & 222 & 255 & 255 & 255 & 233 \\ 115 & 95 & 172 & 219 & 226 & 217 & 152 \\ 108 & 61 & 93 & 139 & 149 & 139 & 97 \\ 101 & 58 & 71 & 93 & 94 & 95 & 111 \\ 89 & 54 & 64 & 60 & 45 & 61 & 158 \\ 64 & 25 & 37 & 22 & 23 & 95 & 193\end{bmatrix}$$

$I_x = \frac{I_1(x+1,y) - I_1(x-1,y)}{2}$:

$$\begin{bmatrix}10.5 & 48.0 & 51.5 & 11.5 & 0.0 & 1.5 & -23.0 \\ 3.5 & 52.5 & 62.5 & 16.5 & 0.0 & -11.0 & -46.5 \\ -13.5 & 28.5 & 62.0 & 27.0 & -1.0 & -37.0 & -43.0 \\ -27.5 & -7.5 & 39.0 & 28.0 & 0.0 & -26.0 & 7.5 \\ -27.0 & -15.0 & 17.5 & 11.5 & 1.0 & 8.5 & 49.0 \\ -27.5 & -12.5 & 3.0 & -9.5 & 0.5 & 56.5 & 67.5 \\ -43.5 & -13.5 & -1.5 & -7.0 & 36.5 & 85.0 & 49.0\end{bmatrix}$$

$I_y = \frac{I_1(x,y+1) - I_1(x,y-1)}{2}$:

$$\begin{bmatrix}-7.5 & -9.5 & -3.5 & 0.0 & 5.0 & 0.5 & -10.0 \\ -9.0 & -28.5 & -28.5 & -18.0 & -13.0 & -19.0 & -51.5 \\ -4.5 & -34.5 & -64.5 & -58.0 & -53.0 & -58.0 & -68.0 \\ -7.0 & -18.5 & -50.5 & -63.0 & -66.0 & -61.0 & -20.5 \\ -9.5 & -3.5 & -14.5 & -39.5 & -52.0 & -39.0 & 30.5 \\ -18.5 & -16.5 & -17.0 & -35.5 & -35.5 & 0.0 & 41.0 \\ -9.0 & -14.5 & -3.5 & -14.5 & 4.0 & 57.0 & 15.0\end{bmatrix}$$

$I_t = I_2(x,y) - I_1(x,y)$ at $d = 0$:

$$\begin{bmatrix}-8 & -23 & -99 & -106 & -23 & 0 & -1 \\ 1 & -7 & -106 & -133 & -41 & -1 & 20 \\ 1 & 26 & -58 & -133 & -75 & -18 & 55 \\ 6 & 54 & 14 & -80 & -65 & -14 & 37 \\ 12 & 53 & 29 & -34 & -23 & -9 & -29 \\ 25 & 56 & 18 & -13 & 10 & -13 & -121 \\ 51 & 90 & 29 & 2 & 17 & -72 & -168\end{bmatrix}$$

Structure tensor (sums over the window):

$$G = \begin{bmatrix}\sum I_x^2 & \sum I_xI_y\\ \sum I_xI_y & \sum I_y^2\end{bmatrix} = \begin{bmatrix}53307.8 & 6062.0 \\ 6062.0 & 56572.2\end{bmatrix},\qquad \lambda_{min} = 48662.1,\ \lambda_{max} = 61217.9$$

Both eigenvalues are well above zero, so $G$ is invertible and the point is trackable (not an edge or a flat patch — the aperture problem does not apply here).

Iterations: $e = I_1(\mathbf{x}) - I_2(\mathbf{x} + \mathbf{d})$ (with $I_2$ sampled by bilinear interpolation), $\mathbf{b} = \sum [I_x e,\ I_y e]^T$, $\Delta\mathbf{d} = G^{-1}\mathbf{b}$, $\mathbf{d} \leftarrow \mathbf{d} + \Delta\mathbf{d}$.

| k | b | Δd | d after step | SSD before step |
|---|---|---|---|---|
| 1 | (57143.5, -11445.5) | (1.108, -0.321) | (1.108, -0.321) | 170067 |
| 2 | (41315.5, 5450.9) | (0.774, 0.013) | (1.882, -0.308) | 54752 |
| 3 | (4901.4, 2994.6) | (0.087, 0.044) | (1.969, -0.264) | 2429 |
| 4 | (572.6, 251.7) | (0.010, 0.003) | (1.979, -0.261) | 1078 |
| 5 | (59.1, 23.1) | (0.001, 0.000) | (1.980, -0.260) | 1011 |

Final SSD over the window: 1005 (was 170067 at d = 0).

**Predicted** position in frame 875: (207.00, 324.00) + (1.980, -0.260) = **(208.98, 323.74)**.  

**Actual** position (patch correlation, independent of LK): **(208.98, 323.90)**. Difference: 0.16 px.
