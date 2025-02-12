# %%
import kagglehub
import os

# %%
kagglehub.dataset_upload("tomoon33/czii-src", "/workspace/src")

# %%
# kagglehub.dataset_upload("tomoon33/czii-src-2nd-stage", "/workspace/src_2nd_stage")
# zip_command = "zip -r /workspace/src_2nd_stage.zip /workspace/src_2nd_stage"
# os.system(zip_command)
# %%
kagglehub.dataset_upload("tomoon33/czii-configs", "/workspace/configs")

# %%
kagglehub.dataset_upload("tomoon33/czii-scripts", "/workspace/scripts")

# %%
kagglehub.dataset_upload(
    "tomoon33/2nd-stage-feature-generator-v1",
    "/workspace/logs/2nd_stage_feature/feature_generator_v1.pkl",
)

kagglehub.dataset_upload(
    "tomoon33/2nd-stage-gbdt-models-v1",
    "/workspace/logs/2nd_stage_model/gbdt_models_v1.pkl",
)


# %%
kagglehub.dataset_upload("tomoon33/czii-pip_packages_v2", "/workspace/pip_packages_v2")

# %%
kagglehub.dataset_upload(
    "tomoon33/czii-copick-config",
    "/workspace/data/czii-cryo-et-object-identification/copick_sub.config",
)
