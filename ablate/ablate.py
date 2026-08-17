'''
The model is loaded as well as the checkpoint 
Top channels.npy is loaded(from probe.py)
Register ablation hook on layer 4 to zero out the top 50 channels
Run full evaluationengine which runs query against gallery and computes mAP and CMC.

'''

import sys
import os
# import file finder for folders that exist outside this folder ex./content/reid-shortcut-audit',
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#sys path runs before any import that depends on it
import config
import torch
import numpy as np
from torchreid.reid.models import build_model
from torchreid.reid.data import ImageDataManager
from torchreid.reid.optim import build_optimizer
from torchreid.reid.optim import build_lr_scheduler
from torchreid.reid.engine import ImageTripletEngine
# validate paths 
if os.path.exists(config.DATASET_ROOT):
	print('DATASET PATH FOUND')
else:
	sys.exit("DATASET PATH IS MISSING")

if os.path.exists(config.CHECKPOINT_PATH):
	print('CHECKPOINT_PATH found')
else:
	print('CHECKPOINT_PATH is missing ')

# to manage test images during the test run	
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

# load top channels
top_channels = np.load(os.path.join(config.OUTPUT_DIR, 'top_channels.npy'))

# ablation hook
# for the top channels we want to zero out their activations
def ablation_hook(module,input,output):
	output[: ,top_channels,:,:]=0 #(batch,2048,7,7)
	return output # returns the changed zeroed tensor for downstream carry

# register hook on layer4
model.layer4.register_forward_hook(ablation_hook)


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
# engine 
engine =ImageTripletEngine(
  datamanager,
  model,
  optimizer,
  scheduler=scheduler,
  use_gpu=True,
)

print("=== Ablation Evaluation ===")
print(f"Zeroing top 10 color-predictive channels in layer4")
print(f"Baseline: mAP 65.3%, Rank-1 82.8%")
print(f"Running evaluation...")

# engine.run with test_only=True
engine.run (
  save_dir=config.OUTPUT_DIR,
  max_epoch=config.EPOCH,
  eval_freq = 60 ,
  test_only=True
)



