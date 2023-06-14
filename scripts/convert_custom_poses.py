import numpy as np
import click
from os.path import join as pjoin


@click.command()
@click.option('--data_dir', type=str, default='.')
def main(data_dir):
    # Number of images
    n_cams = 10

    # Camera to world poses
    # OpenGL style. i.e., `negative z-axis` for looking-to, 'y-axis' for looking-up
    poses = np.zeros([n_cams, 3, 4])
    poses[:, :3, :3] = np.eye(3)                # Rotation
    poses[:, :3, 3] = np.array([0., 0., 0.])    # Translation (camera position)

    # Camera intrinsic parameters.
    intri = np.zeros([n_cams, 3, 3])
    intri[:, 0, 0] = 256.                       # fx
    intri[:, 1, 1] = 256.                       # fy
    intri[:, 0, 2] = 256.                       # cx
    intri[:, 1, 2] = 256.                       # cy
    intri[:, 2, 2] = 1.

    # Camera distortion parameters [k1, k2, p1, p2]
    # See https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    # If the images have been already undistorted, just set them as zero
    distortion_params = np.zeros([n_cams, 4])

    # Near and far bounds of each camera along the z-axis
    bounds = np.zeros([n_cams, 2])
    bounds[:, 0] = 1.                           # Near
    bounds[:, 1] = 100.                         # Far

    data = np.concatenate([
        poses.reshape(n_cams, 12),
        intri.reshape(n_cams, 9),
        distortion_params.reshape(n_cams, 4),
        bounds.reshape(n_cams, 2)
    ], -1)

    print(data.shape)

    # Should be float64 data type
    data = np.ascontiguousarray(np.array(data).astype(np.float64))
    np.save(pjoin(data_dir, 'cams_meta.npy'), data)


if __name__ == '__main__':
    main()
