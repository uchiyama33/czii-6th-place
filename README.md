# CZII - CryoET Object Identification: 6th Place

## Environment
This solution uses the following Docker image:
 - gcr.io/kaggle-gpu-images/python:v153

All required Python packages are listed in the `pip_packages/requirements.txt` file.

## Data
- Place the competition dataset in the `data/` directory.
- Run the data preparation script:
  ```
  cd src
  python prepare_data.py
  ```
- Create simulated data with the following commands
  - The three simulation settings are stored under `polnet/output`.
  - Move the created simulated data under the `data/` directory.
  ```
  cd polnet/gui
  python gen_dataset_for_czii.py
  ```

## Training Models
  ```
  bash train.sh
  ```

<!-- ## Trained Weights
[https://www.kaggle.com/datasets/tomoon33/isic2024-training-logs/](https://www.kaggle.com/datasets/tomoon33/isic2024-training-logs/) -->

## Tuning Threshholds and Analysis of Results in CV
`notebooks/tune_thresh.py`

## Inference

The submission notebook is `czii-submit.ipynb`.

## Acknowledgements
We extend our sincere gratitude to the creators of the following repositories and notebooks, whose outstanding work significantly contributed to our project:

 - [ashleve/lightning-hydra-template: PyTorch Lightning + Hydra. A very user-friendly template for ML experimentation. ⚡🔥⚡](https://github.com/ashleve/lightning-hydra-template)
  - [anmartinezs/polnet: Generates synthetic datasets for Cryo-Electron Tomography](https://github.com/anmartinezs/polnet)
  - [Project-MONAI/MONAI: AI Toolkit for Healthcare Imaging](https://github.com/Project-MONAI/MONAI)
  - [3d-unet using 2d image encoder](https://www.kaggle.com/code/hengck23/3d-unet-using-2d-image-encoder/notebook)