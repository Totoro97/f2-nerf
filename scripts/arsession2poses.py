import os
import json
import numpy as np
import camera_utils


class DataSet:

    def __init__(self, ) -> None:
        pass

    def export(self, sess_path, out_path):

        def proc(x):
            return np.ascontiguousarray(np.array(x).astype(np.float64))

        poses_and_images = f"{sess_path}/poses_and_images"
        images_path = f"{poses_and_images}/images"
        space_json = f"{poses_and_images}/space.json"

        im_names = os.listdir(images_path)
        im_names.sort()
        num_image = len(im_names)

        with open(space_json, "r") as f:
            metas = json.load(f)
        extrs = metas["frames"]
        frame_infos = {}
        for info in extrs:
            frame_infos[os.path.basename(info["file_path"])] = info

        # intrinsic
        cx, cy, fx, fy = metas["cx"], metas["cy"], metas["fl_x"], metas["fl_y"]
        cam2pix = camera_utils.intrinsic_matrix(fx, fy, cx, cy)
        intrs = np.tile(cam2pix[None], (num_image, 1, 1))

        # distortion
        distortion_params = np.zeros([num_image, 4])

        # extrinsic & bounds
        poses = []
        bounds = []

        for name in im_names:
            info = frame_infos[name]
            poses.append(np.asarray(info["transform_matrix"])[:3, :4])
            bounds.append([info["dmin"], info["dmax"]])

        poses = proc(poses)
        intrs = proc(intrs)
        distortion_params = proc(distortion_params)
        bounds = proc(bounds)
        data = np.concatenate([
            poses.reshape(num_image, 12),
            intrs.reshape(num_image, 9),
            distortion_params.reshape(num_image, 4),
            bounds.reshape(num_image, 2)
        ], -1)

        data = proc(data)

        save_path = f"{out_path}/{os.path.basename(sess_path)}"
        os.makedirs(save_path, exist_ok=True)
        np.save(f"{save_path}/cams_meta.npy", data)

        os.system(f"cp -rf {images_path} {save_path}/ ")


if __name__ == "__main__":
    sess_path = "/mnt/nas/share-all/caizebin/03.dataset/nerf/data/arsession/20230828T164146+0800_Capture_OPPO_PDEM30_molly0828"
    out_path = "/mnt/nas/share-all/caizebin/03.dataset/nerf/data/f2nerf"
    dataset = DataSet()
    dataset.export(sess_path, out_path)
