_base_ = "../default.py"

expname = "dvgo_hotdog"
basedir = "./logs/nerf_synthetic"

data = dict(
    # datadir="/content/hw4_data/hw4_data/hotdog",
    dataset_type="blender_test",
    white_bkgd=True,
)
