import click
import numpy as np
from os.path import join as pjoin


@click.command()
@click.option('--data_dir', type=str)
def hello(data_dir):
    poses_bounds = np.load(pjoin(data_dir, 'poses_bounds.npy')).reshape(-1, 17)
    poses_hwf = poses_bounds[:, :15].reshape(-1, 3, 5)
    poses = poses_hwf[:, :3, :4]
    hwf = poses_hwf[:, :3, 4]
    poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], 2)
    bounds = poses_bounds[:, 15: 17]
    n_poses = len(poses)
    intri = np.zeros([n_poses, 3, 3])
    intri[:, :3, :3] = np.eye(3)
    intri[:, 0, 0] = hwf[:, 2]
    intri[:, 1, 1] = hwf[:, 2]
    intri[:, 0, 2] = hwf[:, 1] * .5
    intri[:, 1, 2] = hwf[:, 0] * .5

    data = np.concatenate([
        poses.reshape(n_poses, -1),
        intri.reshape(n_poses, -1),
        np.zeros([n_poses, 4]),
        bounds.reshape(n_poses, -1)
    ], -1)

    data = np.ascontiguousarray(np.array(data).astype(np.float64))
    np.save(pjoin(data_dir, 'cams_meta.npy'), data)


if __name__ == '__main__':
    hello()
