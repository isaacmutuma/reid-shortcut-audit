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

"""Building the model ,
                   loads the checkpoint's weights , 
run images and 
				   extracts activations"""

# validate paths 
if os.path.exists(config.DATASET_ROOT):
	print('DATASET PATH FOUND')
else:
	sys.exit("DATASET PATH IS MISSING")

if os.path.exists(config.CHECKPOINT_PATH):
	print('CHECKPOINT_PATH found')
else:
	print('CHECKPOINT_PATH is missing ')
	


# datamanager to load training images and their labels
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
# the model that we load our trained weights ans extract activations from
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
checkpoint=torch.load(config.CHECKPOINT_PATH,map_location=device)
# EXtract only the layer weights dictionary ignoring the 'optimizer', 'scheduler', 'epoch', 'rank1'
model.load_state_dict(checkpoint['state_dict'])
model.eval()


# capture activation values from the 4 layers
activations = {}
def make_hook(name):
    def capture(module, input, output): # module is the layer itself on the resnet ,input is tensor passed from layer to layer ,output is a the activation map
            activations[name] = output.detach()
    return capture
#whenever your forward pass runs, also call this function
model.layer1.register_forward_hook(make_hook('layer1')) # register receives the function to call it whenever
model.layer2.register_forward_hook(make_hook('layer2'))
model.layer3.register_forward_hook(make_hook('layer3'))
model.layer4.register_forward_hook(make_hook('layer4'))

#evaluating on extraced activation values
vectors = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}
vector_pids = []
for batch in datamanager.train_loader: #dataloader implements __iter__ and __next__ to help iterate over objects
    images = batch[0].to(device)
    pids = batch[1]
    with torch.no_grad():
        model(images)# activations dict for a batch
        # a 32 image batch will have (32,256,56,56), (32, 512, 28, 28),(32, 1024, 14, 14),(32, 2048, 7, 7) per layer
        #each image passes through all 4 layers so need to store each images activation values in a dict 
        #flatten a layers' activation per image and store it a vector with key as pid
        for name in  ['layer1', 'layer2', 'layer3', 'layer4']:
            layer_vectors=activations[name].mean(dim=[2,3]) # (32, 256, 56, 56) → (32, 256) for layer 1 (32, 512)layer 2 (32, 1024,) layer 3 layer 4 (32, 2048,)
            vectors[name].append(layer_vectors.cpu().numpy())
        vector_pids.extend(pids.numpy())
    
for name in ['layer1', 'layer2', 'layer3', 'layer4']:
    vectors[name] = np.concatenate(vectors[name], axis=0)  #(32,256)>>(64,256)>>(96,256)
vector_pids = np.array(vector_pids)      
 
 # per image we have a vector in every single layer ex layer one is (12936,256) for all training images we have a vector 256 elements




    

