# 1. Prepare running environment

- step 1: start container
    - bash artifacts/create_container.sh 
- step 2: package installation
    - upgrade
        - cmake==3.22.4
        - libtorch: refer to README.md
    - required python package
        - pip install -r requirements.txt
- step 3: build
    - fix warning
        - add `find_package(PythonInterp REQUIRED)` to shield warning
    - build
        - cmake . -B build
        - cmake --build build --target main --config RelWithDebInfo -j

# 2. Training ARSession Dataset

- step 1: Convert ARsession Dataset to F2Nerf Dataset
```
python scripts/arsession2poses.py
```

- step 2: training, modify the dataset info in run_arsess01.sh and excute the script
```
bash run_arsess01.sh
```