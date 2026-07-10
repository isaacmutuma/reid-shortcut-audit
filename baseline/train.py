from pickle import TRUE
import config
import os
import sys
import torchreid
from torchreid.utils.data import ImageDataManager
from torchreid.data import Market1501
from ..dataset import ImageDataset



# check whether the file path exists and is accurate
if os.path.exists(config.DATASET_ROOT):
	print('Path to Dataset is found')
else:
	sys.exit('Path to Dataset is missing!')


# check whether the output path exists
if os.path.exists(config.OUTPUT_DIR):
	print('OUTPUT_DIR created')
else:
	os.makedirs(config.OUTPUT_DIR,exist_ok=True)

# load dataset - triggers the __init__ funtion to run ans pass three tuples to the superclass
dataset = Market1501(root=config.DATASET_ROOT)

# dataloader
datamanager = ImageDataManager(
    root=config.DATASET_ROOT,
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    transforms='random_flip',
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    batch_size_train=config.BATCH_SIZE,
    batch_size_test=config.BATCH_SIZE,
    workers=4,
    use_gpu=True
)

