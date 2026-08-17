'''The stress test is essentially asking: 
"does this model track identity or does it track shirts?"

If the model passes — it finds the correct person despite the shirt color change.
Identity features (body shape, proportions, gait patterns) are strong enough to overcome the color mismatch. 
The model is robust.

If the model fails — it returns people wearing similar colored shirts to the original, not the correct person.
The color shortcut is so dominant that changing the shirt color breaks the match entirely.
That failure rate is your severity score.'''
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
import torch
import config
from torchreid.reid.data import ImageDataManager
from torchreid.reid.models import build_model
from torchreid.reid.optim import build_optimizer
from torchreid.reid.engine import ImageTripletEngine
from torchreid.reid.optim import build_lr_scheduler
from torch.utils.data import Dataset, DataLoader
from PIL import Image

#query_data = datamanager.test_dataset['market1501']['query']  # list of tuples from process dir

class ColorShiftDataset(Dataset):
	"""Wraps any torchreid query dataset and applies
	controllable HSV hue shift to the torso region.
	Enables appearance-shift stress testing without
	requiring separate clothing-change datasets."""

	def __init__(self, query_data, hue_shift, transform=None):
		"""
		Args:
			query_data: list of (img_path, pid, camid) tuples
			hue_shift:  int, hue shift in degrees (0-180 in OpenCV HSV)
			transform:  torchvision transforms to apply after recoloring
		"""
		self.query_data=query_data
		self.hue_shift = hue_shift
		self.transform = transform

	
	def __len__(self):
		# Return the total number of query images
		return len(self.query_data)

	def __getitem__(self, index):
		# unpacks a query images data
		img_path, pid, camid, dsetid = self.query_data[index]
		# load image
		orig_img = cv2.imread(img_path)
		if orig_img is None:
			raise FileNotFoundError(f"Image not found at: {img_path}")
		# Create a copy
		img = orig_img.copy()
		# torso crop
		torso=img[70:170, :]
		# convert to HSV
		hsv_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
		#extract the hsv channel
		h_channel = hsv_torso[:, :, 0]
		# shift the hsv channel by hue_shift
		h_channel=(h_channel + self.hue_shift) % 180
		#plug in the channel into the torso
		hsv_torso[:, :, 0] = h_channel
		#convert the torso back to BGR for tranform pipeline
		bgr_torso=cv2.cvtColor(hsv_torso,cv2.COLOR_HSV2BGR)
		# bgr_torso back to original torso_image
		img[70:170, :] = bgr_torso
		#transform expects PIL image, not BGR numpy array
		img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		# apply tranforms to convert to a tensor normalize etc
		if self.transform is not None:
			img_tensor=self.transform(img_pil)
		else:
			img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


		return {
					'img':    img_tensor,
					'pid':    pid,
					'camid':  camid,
					'impath': img_path
				}


# check whether the file path exists and is accurate
if os.path.exists(config.DATASET_ROOT):
	print('Path to Dataset is found')
else:
	sys.exit('Path to Dataset is missing!')


# check whether the output path exists
if not os.path.exists(config.OUTPUT_DIR):
	os.makedirs(config.OUTPUT_DIR,exist_ok=True)
	print('OUTPUT_DIR created')
else:
	print('OUTPUT_DIR found ')
	

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
    workers=2,
    use_gpu=True
)
model = build_model(
   name=config.MODEL,
   num_classes=datamanager.num_train_pids, # internal access soley for datamanager to accees this data on unique training data 
   loss='triplet', # outputs, features = self.model(imgs)
   pretrained = True,
   use_gpu=True    
)

# model to device
device=torch.device(config.DEVICE)
model=model.to(device)

#load the checkpoint
checkpoint=torch.load(config.CHECKPOINT_PATH,map_location=device,weights_only=False)
# EXtract only the layer weights dictionary ignoring the 'optimizer', 'scheduler', 'epoch', 'rank1'
model.load_state_dict(checkpoint['state_dict'])
model.eval()


#optimizer 
optimizer = build_optimizer(
   model,
   optim='adam',
   lr=config.LR
)
# to adjust the learning rates in the training process
scheduler=build_lr_scheduler(
  optimizer,
  lr_scheduler='single_step',
  stepsize=20,
  gamma=0.1,
  max_epoch=1
)

#dual loss architecture using the triplet engine
engine =ImageTripletEngine(
  datamanager,
  model,
  optimizer,
  scheduler=scheduler,
  use_gpu=True,
)

# get query data and build recolored dataset
query_data = datamanager.test_dataset['market1501']['query']

dataset = ColorShiftDataset(
    query_data=query_data,
    hue_shift=180,
    transform=datamanager.transform_te
)

recolored_loader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

# swap query loader and evaluate
engine.test_loader['market1501']['query'] = recolored_loader

print("=== Stress Test: Hue Shift 180 degrees ===")
print("Baseline: mAP 65.3%, Rank-1 82.8%")
engine.test(save_dir=config.OUTPUT_DIR)






