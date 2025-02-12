FROM gcr.io/kaggle-gpu-images/python:v153

COPY kaggle.json /root/.kaggle/kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json

# --------- pytorch --------- #
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install lightning>=2.0.0
RUN pip install torchmetrics>=0.11.4

# --------- hydra --------- #
RUN pip install hydra-core==1.3.2
RUN pip install hydra-colorlog==1.2.0
RUN pip install hydra-optuna-sweeper==1.2.0

# --------- others --------- #
RUN pip install rootutils       # standardizing the project root setup
RUN pip install pre-commit      # hooks for applying linters on commit
RUN pip install rich            # beautiful text formatting in terminal
RUN pip install pytest          # tests

# czii
RUN pip install napari monai zarr "numpy<2" cryoet-data-portal==4.0.0 s3fs trimesh ome-zarr scikit-image trimesh pyside2
RUN pip install numcodecs connected-components-3d copick git+https://github.com/copick/copick-utils.git

RUN pip install tensorrt-cu12==10.3.0
RUN pip install torch-tensorrt==2.5.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install einops segmentation-models-pytorch
RUN pip install timm-3d timm==1.0.12