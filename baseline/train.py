import sys
import os
# import file finder for folders that exist outside this folder ex./content/reid-shortcut-audit',
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#sys path runs before any import that depends on it
import config
import torch
from torchreid.reid.data import ImageDataManager
from torchreid.reid.models import build_model
from torchreid.reid.optim import build_optimizer
from torchreid.reid.engine import ImageTripletEngine

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
	

"""ImageDataManager
  → calls init_image_dataset('market1501')
    → looks up registry
      → finds Market1501 class
        → instantiates Market1501(root=...)
          → __init__ runs
            → process_dir runs x3
              → three tuple lists built
                → passed to super().__init__()"""
# The manager then  wraps the tuples from the dataset object in dataloaders
# transforms and images ready to be passed it to network 
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
# the model instance passed to the source code
''' _num_train_pids    ← discovered and stored internally
    num_train_pids   ← property that exposes it externally'''
model = build_model(
   name=config.MODEL,
   num_classes=datamanager.num_train_pids, # internal access soley for datamanager to accees this data on unique training data 
   loss='softmax_triplet',
	 pretrained = True,
   use_gpu=True    
    )

# move the model to device
device = torch.device(config.DEVICE) # the actual hardware object that pytorch understands torch.device()
model=model.to(device)

#optimizer 
optimizer = build_optimizer(
   model,
   optim='adam',
   lr=config.LR
)

#dual loss architecture using the triplet engine
engine =ImageTripletEngine(
  datamanager,
  model,
  optimizer,
  use_gpu=True,
)
# running the engine just needs to know where to save results how long to train and how frequent to evaluate
engine.run (
  save_dir=config.OUTPUT_DIR,
  max_epoch=config.EPOCH,
  eval_freq = 5
  )


         

      


