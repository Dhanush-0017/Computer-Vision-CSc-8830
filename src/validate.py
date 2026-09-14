"""
validate.py -- Step 3, checking how accurate Step 2 actually is.

Takes the 20 rows in measurements.csv (each one a pixel pair + the Z I
measured + the real length from a tape measure) and runs them all through
measure(), then compares against the ground truth to get error stats.

measurements.csv columns:
    id, Z_mm, u1, v1, u2, v2, gt_mm
        Z_mm  : camera-to-object distance, tape measured, mm (>2000 per the
                assignment spec)
        u1,v1 / u2,v2 : the two clicked pixel points
        gt_mm : the true length between them, also tape measured

Prints MAE, RMSE, MAPE, mean/std of the signed error, and max error, and
writes validation_results.csv. --plot also saves a measured-vs-ground-truth
scatter and an error-vs-distance plot, mostly because "error grows with
distance" is easier to show than to explain in text.

Run:
    python validate.py --calib ../calibration/camera_params.npz \
                       --csv ../data/measurements.csv --plot
"""
import argparse
import numpy as np
import pandas as pd
from measure import load_calib, measure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()

    K, dist = load_calib(args.calib)
    df = pd.read_csv(args.csv)

    est, err, pct = [], [], []
    for _, r in df.iterrows():
        d = measure((r.u1, r.v1), (r.u2, r.v2), r.Z_mm, K, dist)
        est.append(d)
        err.append(d - r.gt_mm)
        pct.append(100.0 * (d - r.gt_mm) / r.gt_mm)

    df['est_mm'] = np.round(est, 2)
    df['error_mm'] = np.round(err, 2)
    df['error_pct'] = np.round(pct, 2)

    err = np.array(err)
    pct = np.array(pct)
    stats = {
        'N': len(df),
        'MAE_mm': float(np.mean(np.abs(err))),
        'RMSE_mm': float(np.sqrt(np.mean(err ** 2))),
        'Mean_signed_error_mm': float(np.mean(err)),
        'Std_error_mm': float(np.std(err, ddof=1)) if len(df) > 1 else 0.0,
        'MAPE_pct': float(np.mean(np.abs(pct))),
        'Max_abs_error_mm': float(np.max(np.abs(err))),
    }

    print(df.to_string(index=False))
    print('\n=== Error statistics ===')
    for k, v in stats.items():
        print(f'{k:24s}: {v:.3f}' if isinstance(v, float) else f'{k:24s}: {v}')

    df.to_csv('validation_results.csv', index=False)
    print('\nSaved validation_results.csv')

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.scatter(df.gt_mm, df.est_mm)
        lim = [min(df.gt_mm.min(), df.est_mm.min()),
               max(df.gt_mm.max(), df.est_mm.max())]
        plt.plot(lim, lim, 'r--', label='ideal (y=x)')
        plt.xlabel('Ground truth (mm)')
        plt.ylabel('Estimated (mm)')
        plt.title('Measured vs Ground Truth')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('validation_plot.png', dpi=150)
        print('Saved validation_plot.png')


if __name__ == '__main__':
    main()
