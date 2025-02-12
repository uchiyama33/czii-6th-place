# CZII - CryoET Object Identification: 6th Place

## Environment
This solution was developed using the following Docker image:
 - gcr.io/kaggle-gpu-images/python:v153

All required Python packages can be found in `pip_packages/requirements.txt`.

## Data
- Place the competition dataset in the `data/` directory.
- Run the data preparation script:
  ```
  cd src
  python prepare_data.py
  ```
- Create simulated data using the following commands
  - The three simulation settings are stored under `polnet/output`.
  - Move the generated simulated data under the `data/` directory.
  ```
  cd polnet/gui
  python gen_dataset_for_czii.py
  ```

## Training Models
  ```
  bash train.sh
  ```

## Tuning Threshholds and Analysis of Results in CV
Use the `notebooks/tune_thresh.py` script to tune thresholds and analyze cross-validation results.

## Inference
- The sample notebook that converts a PyTorch model to a TensorRT engine is `czii-convert-trt-250101-hard-r05-ftp-pre-1221-env2b2.ipynb`.
- The final submission is generated in the `czii-submission-6th-place.ipynb` notebook.

## Acknowledgements
We extend our sincere gratitude to the creators of the following repositories and notebooks, whose outstanding work significantly contributed to our project:

 - [ashleve/lightning-hydra-template: PyTorch Lightning + Hydra. A very user-friendly template for ML experimentation. ⚡🔥⚡](https://github.com/ashleve/lightning-hydra-template)
  - [anmartinezs/polnet: Generates synthetic datasets for Cryo-Electron Tomography](https://github.com/anmartinezs/polnet)
  - [Project-MONAI/MONAI: AI Toolkit for Healthcare Imaging](https://github.com/Project-MONAI/MONAI)
  - [3d-unet using 2d image encoder](https://www.kaggle.com/code/hengck23/3d-unet-using-2d-image-encoder/notebook)