# F2-NeRF
This is the repo for the implementation of **F2-NeRF: Fast Neural Radiance Field Training with Free Camera Trajectories**.

![](./static/intro_1.gif)
![](./static/intro_2.gif)

## [Project page](https://totoro97.github.io/projects/f2-nerf) |  [Paper](https://arxiv.org/abs/2303.15951) | [Data](https://www.dropbox.com/sh/jmfao2c4dp9usji/AAC7Ydj6rrrhy1-VvlAVjyE_a?dl=0)


## Install
The development of this project is primarily based on LibTorch.
### Step 1. Install dependencies

For Debian based Linux distributions:
```
sudo apt install zlib1g-dev
```

For Arch based Linux distributions:
```
sudo pacman -S zlib
```


### Step 2. Clone this repository

```shell
git clone --recursive https://github.com/Totoro97/f2-nerf.git
cd f2-nerf
```

### Step 3. Download pre-compiled LibTorch
We take `torch-1.13.1+cu117` for example.
```shell
cd External
wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu117.zip
unzip ./libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu117.zip
```

### Step 4. Compile
The lowest g++ version I have tested is 7.5.0.
```shell
cd ..
cmake . -B build
cmake --build build --target main --config RelWithDebInfo -j
```

## Run

### Training
Here is an example command to train F2-NeRF:
```shell
python scripts/run.py --config-name=wanjinyou dataset_name=example case_name=ngp_fox mode=train +work_dir=$(pwd)
```

### Render test images
Simply run:
```shell
python scripts/run.py --config-name=wanjinyou dataset_name=example case_name=ngp_fox mode=test is_continue=true +work_dir=$(pwd)
```

### Render path
We provide a script to generate render path (by interpolating the input camera poses). For example, for the fox data, run:

```shell
python scripts/inter_poses.py --data_dir ./data/example/ngp_fox --key_poses 5,10,15,20,25,30,35,40,45,49 --n_out_poses 200
```

The file `poses_render.npy` in the data directory would be generated. Then run

```shell
python scripts/run.py --config-name=wanjinyou dataset_name=example case_name=ngp_fox mode=render_path is_continue=true +work_dir=$(pwd)
```

The synthesized images can be found in `./exp/ngp_fox/test/novel_images`.

## Train F2-NeRF on your custom data
Make sure the images are at `./data/<your-dataset-name>/<your-case-name>/images`
1. Run COLMAP SfM:
```shell
bash scripts/local_colmap_and_resize.sh ./data/<your-dataset-name>/<your-case-name>
```
or run [hloc](https://github.com/cvg/Hierarchical-Localization) if COLMAP failed. (Make sure [hloc](https://github.com/cvg/Hierarchical-Localization) has been installed)
```shell
bash scripts/local_hloc_and_resize.sh ./data/<your-dataset-name>/<your-case-name>
```

2. Generate cameras file:
```shell
python scripts/colmap2poses.py --data_dir ./data/<your-dataset-name>/<your-case-name>
```

3. Run F2-NeRF using the similar command as in the example data:
```shell
python scripts/run.py --config-name=wanjinyou \
dataset_name=<your-dataset-name> case_name=<your-case-name> mode=train \
+work_dir=$(pwd)
```

## Train F2-NeRF on LLFF/NeRF-360-V2 dataset
We provide a script to convert the LLFF camera format to our camera format. For example:
```
python scripts/llff2poses.py --data_dir=xxx/nerf_llff_data/horns
```

## TODO/Future work
- Add anti-aliasing

## Acknowledgment
Besides LibTorch, this project is also built upon the following awesome libraries:
- [happly](https://github.com/nmwsharp/happly) for I/O of PLY files
- [stb_image](https://github.com/nothings/stb) for I/O of image files
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for fast MLP training/inference
- [eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) for linear algebra computing
- [yaml-cpp](https://github.com/jbeder/yaml-cpp) for I/O of YAML files
- [hydra](https://github.com/facebookresearch/hydra) for configuration
- [cnpy](https://github.com/rogersce/cnpy) for I/O of npy files

Some of the code snippets are inspired from [instant-ngp](https://github.com/NVlabs/instant-ngp), [torch-ngp](https://github.com/ashawkey/torch-ngp) and [ngp-pl](https://github.com/kwea123/ngp_pl).
The COLMAP processing scripts are from [multinerf](https://github.com/google-research/multinerf). The example data `ngp_fox` is from  [instant-ngp](https://github.com/NVlabs/instant-ngp).

## Citation
Cite as below if you find this repository is helpful to your project:

```
@article{wang2023f2nerf,
  title={F2-NeRF: Fast Neural Radiance Field Training with Free Camera Trajectories},
  author={Wang, Peng and Liu, Yuan and Chen, Zhaoxi and Liu, Lingjie and Liu, Ziwei and Komura, Taku and Theobalt, Christian and Wang, Wenping},
  journal={CVPR},
  year={2023}
}
```
