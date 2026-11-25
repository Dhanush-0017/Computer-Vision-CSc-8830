"""
measure_stereo.py -- Problem 1 from the command line.

Measures the three example objects (notebook, cream tube, lid) from their
two photos and prints the result next to the ruler measurement.

The clicked points for each object are saved in data/examples.json, so this
script gives the same numbers as the web page.

Run (from inside modules/module7):
    python measure_stereo.py
    python measure_stereo.py --baseline 50        # slide distance in mm
"""
import argparse
import json
import os

import cv2

import stereo

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'photos')   # Problem 1 photo pairs


def percent_error(estimate, truth):
    return 100.0 * (estimate - truth) / truth


def measure_example(name, example, baseline_mm):
    left = cv2.imread(os.path.join(DATA, example['left']), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(os.path.join(DATA, example['right']), cv2.IMREAD_GRAYSCALE)
    height, width = left.shape

    # step 1: camera from the Module 2 calibration
    K, dist = stereo.load_camera(width, height)

    # step 2: how the phone moved between the photos
    R, t, pts_left, pts_right = stereo.find_camera_motion(left, right, K, dist,
                                                          baseline_mm)
    F = stereo.fundamental_from_pose(K, R, t)
    checks = stereo.matrix_checks(K, R, t)

    print('=' * 60)
    print(name)
    print('  matched features used:', len(pts_left))
    angles = stereo.rotation_in_degrees(R)
    print('  phone turned by (deg): x %.1f, y %.1f, z %.1f'
          % (angles[0], angles[1], angles[2]))
    sv = checks['E_singular_values']
    print('  checks: rank(F) = %d, singular values of E = (%.3f, %.3f, %.1e), '
          'det(R) = %.6f'
          % (checks['F_rank'], sv[0], sv[1], sv[2], checks['det_R']))

    # steps 3 and 4: match every clicked point and turn it into 3D
    points_3d = []
    for click in example['clicks']:
        match, score = stereo.find_match(left, right, click, F)
        X = stereo.triangulate(click, match, K, dist, R, t)
        points_3d.append(X)
        print('  point (%4d, %4d) -> right (%6.1f, %6.1f)  match %.2f  Z = %.1f cm'
              % (click[0], click[1], match[0], match[1], score, X[2] / 10))

    # step 5: sizes
    truth = example['true_mm']
    if example['shape'] == 'rectangle':
        width_mm, height_mm, edges = stereo.rectangle_size(points_3d)
        print('  width : %6.1f mm  (ruler %d mm, error %+.1f%%)'
              % (width_mm, truth['width'], percent_error(width_mm, truth['width'])))
        print('  height: %6.1f mm  (ruler %d mm, error %+.1f%%)'
              % (height_mm, truth['height'], percent_error(height_mm, truth['height'])))

    elif example['shape'] == 'circle':
        diameter, across, down = stereo.circle_diameter(points_3d)
        print('  diameter: %6.1f mm  (ruler %d mm, error %+.1f%%)'
              % (diameter, truth['diameter'],
                 percent_error(diameter, truth['diameter'])))

    else:
        edges = stereo.polygon_edges(points_3d)
        for i in range(len(edges)):
            print('  edge %d: %6.1f mm  (ruler %d mm, error %+.1f%%)'
                  % (i + 1, edges[i], truth['edges'][i],
                     percent_error(edges[i], truth['edges'][i])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=float, default=None,
                        help='slide distance between the two photos in mm '
                             '(default: the value saved for each example)')
    args = parser.parse_args()

    with open(os.path.join(DATA, 'examples.json')) as f:
        examples = json.load(f)

    for name in examples:
        example = examples[name]
        baseline_mm = example['baseline_mm']
        if args.baseline is not None:
            baseline_mm = args.baseline
        measure_example(name, example, baseline_mm)


if __name__ == '__main__':
    main()
