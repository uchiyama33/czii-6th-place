# %%
from core.vtk_utilities import *
from core.utilities import *
from core.widgets_utilities import *
from core.all_features2 import all_features2
from core.tk_utilities import *
import io
import os
from joblib import Parallel, delayed
import numpy as np
import datetime

# %%
rnd = np.random.default_rng(datetime.datetime.now().microsecond)

# %%
# Generate a synthetic dataset
#
# This script starts the process to generate a synthetic dataset taken already created structural models.


output_directory_root = "/home/tomo/kaggle/polnet/output/sim_250111_woCZII"
os.makedirs(output_directory_root, exist_ok=True)

sim_directory = os.path.join(output_directory_root, "simulated")
os.makedirs(sim_directory, exist_ok=True)

# %%
use_pns_list = [
    # "/home/tomo/kaggle/polnet/data/in_10A_czii/1fa2_10A.pns",
    # "/home/tomo/kaggle/polnet/data/in_10A_czii/6cvm_10A.pns",
    # "/home/tomo/kaggle/polnet/data/in_10A_czii/6n4v_10A.pns",
    # "/home/tomo/kaggle/polnet/data/in_10A_czii/6qzp_10A.pns",
    # "/home/tomo/kaggle/polnet/data/in_10A_czii/6z6u_10A.pns",
    # "/home/tomo/kaggle/polnet/data/in_10A_czii/7N4Y_10A.pns",
    "/home/tomo/kaggle/polnet/data/in_10A_czii/8vaf_10A.pns",
    "/home/tomo/kaggle/polnet/data/in_10A/3j9i_10A.pns",
    "/home/tomo/kaggle/polnet/data/in_10A/1u6g_10A.pns",
    "/home/tomo/kaggle/polnet/data/in_10A/2uv8_10A.pns",
    "/home/tomo/kaggle/polnet/data/in_10A/3h84_10A.pns",
    "/home/tomo/kaggle/polnet/data/in_10A/4cr2_10A.pns",
    "/home/tomo/kaggle/polnet/data/in_10A/5mrc_10A.pns",
]

use_mbs_list = [
    "/home/tomo/kaggle/polnet/data/in_mbs/ellipse.mbs",
    # "/home/tomo/kaggle/polnet/data/in_mbs/sphere.mbs",
    # "/home/tomo/kaggle/polnet/data/in_mbs/toroid.mbs",
]

use_hns_list = [
    "/home/tomo/kaggle/polnet/data/in_helix/actin.hns",
    "/home/tomo/kaggle/polnet/data/in_helix/mt.hns",
]

# fixed parameters
ntomos = 30
voi_shape = (630, 630, 200)
x, y, z = voi_shape
voi_off = (
    (int(x * 0.025), int(x * 0.975)),
    (int(y * 0.025), int(y * 0.975)),
    (int(z * 0.025), int(z * 0.975)),
)
voi_size = 10
mmer_tries = 20
pmer_tries = 100
surf_dec = 0.9
tilt_angs = range(-45, 46, 3)
detector_snr = [0.2, 3.0]
malign_mn = 1
malign_mx = 5
malign_sg = 0.5

# Save lists and parameters to file
with open(os.path.join(output_directory_root, "params.txt"), "w") as f:
    f.write(f"use_pns_list = {use_pns_list}\n")
    f.write(f"use_mbs_list = {use_mbs_list}\n")
    f.write(f"use_hns_list = {use_hns_list}\n")
    f.write(f"ntomos = {ntomos}\n")
    f.write(f"voi_shape = {voi_shape}\n")
    f.write(f"voi_off = {voi_off}\n")
    f.write(f"voi_size = {voi_size}\n")
    f.write(f"mmer_tries = {mmer_tries}\n")
    f.write(f"pmer_tries = {pmer_tries}\n")
    f.write(f"surf_dec = {surf_dec}\n")
    f.write(f"tilt_angs = {tilt_angs}\n")
    f.write(f"detector_snr = {detector_snr}\n")
    f.write(f"malign_mn = {malign_mn}\n")
    f.write(f"malign_mx = {malign_mx}\n")
    f.write(f"malign_sg = {malign_sg}\n")


def run_all_features2(i):
    output_directory = os.path.join(sim_directory, f"tomogram_{i}")
    os.makedirs(output_directory, exist_ok=True)

    all_features2(
        1,
        voi_shape,
        output_directory,
        voi_off,
        voi_size,
        mmer_tries,
        pmer_tries,
        use_mbs_list,
        use_hns_list,
        use_pns_list,
        [],
        surf_dec,
        tilt_angs,
        detector_snr,
        malign_mn,
        malign_mx,
        malign_sg,
        random_seed=rnd.integers(0, 1000000),
    )


Parallel(n_jobs=6)(delayed(run_all_features2)(i) for i in range(ntomos))

# %%
# 生成したデータを一つにまとめる
# 対象データ
# - トモグラム：　{output_directory_root}/tomogram_{i}/tomos/tomo_rec_0_snr*.mrc
# - ラベル：　{output_directory_root}/tomogram_{i}/tomos/tomo_lbls_0.mrc
# mrcはnumpyに変換して保存する

import shutil
import os
from glob import glob
import mrcfile


def read_mrc(file_path):
    with mrcfile.open(file_path) as mrc:
        data = mrc.data
    return data


output_directory = os.path.join(output_directory_root, "dataset")
os.makedirs(output_directory, exist_ok=True)

for i in range(ntomos):
    save_directory = f"{output_directory}/tomogram_{i}"
    os.makedirs(save_directory, exist_ok=True)

    tomo_file = glob(f"{sim_directory}/tomogram_{i}/tomos/tomo_rec_0_snr*.mrc")[0]
    tomo = read_mrc(tomo_file)
    np.save(f"{save_directory}/tomo.npy", tomo)

    lbl_file = f"{sim_directory}/tomogram_{i}/tomos/tomo_lbls_0.mrc"
    lbl = read_mrc(lbl_file)
    np.save(f"{save_directory}/label.npy", lbl)

    shutil.copyfile(
        f"{sim_directory}/tomogram_{i}/tomos_motif_list.csv",
        f"{save_directory}/tomos_motif_list.csv",
    )


# あとは/home/tomo/kaggle/polnet/output/241130/tomogram_0/labels_table.csvと
# /home/tomo/kaggle/polnet/output/241130/tomogram_0/tomos_motif_list.csv
# をコピーしてくる
shutil.copyfile(
    f"{sim_directory}/tomogram_0/labels_table.csv",
    f"{output_directory}/labels_table.csv",
)
shutil.copyfile(
    f"{sim_directory}/tomogram_0/tomos_motif_list.csv",
    f"{output_directory}/tomos_motif_list.csv",
)

# %%
