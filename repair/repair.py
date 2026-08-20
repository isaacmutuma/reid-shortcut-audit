'''In this file,
 The shortcut exists because Market-1501's training data has perfect color consistency per identity.
We break that consistency during training,
forcing the model to find features that are consistent despite color variation.'''


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
import torch
import config
import random
from torchreid.reid.data import ImageDataManager
from torchreid.reid.models import build_model
from torchreid.reid.optim import build_optimizer
from torchreid.reid.engine import ImageTripletEngine
from torchreid.reid.optim import build_lr_scheduler
from torch.utils.data import Dataset, DataLoader
from PIL import Image


""" Custom transform to enable torso specific hue shift"""
class TorsoColorJitter():
	"""store the maximum hue shift range:"""
	def __init__(self, max_hue_shift=30): 
		self.max_hue_shift = max_hue_shift
	"""receives a PIL image, applies random torso hue shift, returns PIL image:"""
	def __call__(self,img):
		#PIL RGB → numpy RGB
		img_np = np.array(img)  
		# RGB → BGR
		img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  
		#Crop torso
		img_torso= img_bgr[70:170,:]
		#convert to HSV
		hsv_torso=cv2.cvtColor(img_torso,cv2.COLOR_BGR2HSV)
		#extract the hsv channel
		h_channel = hsv_torso[:, :, 0]
		#shift
		shift = random.randint(-self.max_hue_shift, self.max_hue_shift)
		#apply shift
		h_channel = (h_channel.astype(int) + shift) % 180
		h_channel = h_channel.astype(np.uint8)
		# plug the channel back into the torso
		hsv_torso[:, :, 0] = h_channel
		#convert the torso back to BGR for tranform pipeline
		bgr_torso=cv2.cvtColor(hsv_torso,cv2.COLOR_HSV2BGR)
		# bgr torso back to the image
		img_bgr[70:170, :] = bgr_torso
		#transform expects PIL image, not BGR numpy array
		img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
	
		return img_pil



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
# inject AFTER instantiation
datamanager.transform_tr.transforms.insert(-2, TorsoColorJitter(max_hue_shift=30))
print("TorsoColorJitter injected into training pipeline")

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
  max_epoch=config.EPOCH
)

#dual loss architecture using the triplet engine
engine =ImageTripletEngine(
  datamanager,
  model,
  optimizer,
  scheduler=scheduler,
  use_gpu=True,
)
REPAIR_OUTPUT = os.path.join(config.OUTPUT_DIR, 'repair')
os.makedirs(REPAIR_OUTPUT, exist_ok=True)

engine.run(
    save_dir=REPAIR_OUTPUT,
    max_epoch=config.EPOCH,
    eval_freq=config.EPOCH
)