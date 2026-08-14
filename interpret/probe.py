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
from color_labels import extract_color_labels
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
''' batch['img']    → 32 image tensors  → push to GPU → model(images) → hooks fire
    batch['pid']    → 32 identity labels
    batch['impath'] → 32 image paths    → keys for both vectors and color labels
vectors = {
    'layer1': {'path1': array(256,), 'path2': array(256,), ... × 12936},
    'layer2': {'path1': array(512,), ...},
    'layer3': {'path1': array(1024,), ...},
    'layer4': {'path1': array(2048,), ...}
}
vector_pids = {'path1': 0, 'path2': 0, 'path3': 1, ...}'''
 
train_dir = os.path.join(config.DATASET_ROOT, 'market1501', 'Market-1501-v15.09.15', 'bounding_box_train')
color_labels = extract_color_labels(train_dir)

vectors = {'layer1': {}, 'layer2': {}, 'layer3': {}, 'layer4': {}}
vector_pids = {}
for batch in datamanager.train_loader: #dataloader implements __iter__ and __next__ to help iterate over objects
    images = batch['img'].to(device)
    pids = batch['pid'] #dataloader collects 32 into a batch it stacks them into a PyTorch tensor:
    paths = batch['impath']

    with torch.no_grad():
        model(images)# activations dict for a batch
        # a 32 image batch will have (32,256,56,56), (32, 512, 28, 28),(32, 1024, 14, 14),(32, 2048, 7, 7) per layer
        #each image passes through all 4 layers so need to store each images activation values in a dict 
        #flatten a layers' activation per image and store it a vector with key as pid
    for i,path in enumerate(paths):
        for name in  ['layer1', 'layer2', 'layer3', 'layer4']:
            # (32, 256, 56, 56) → (32, 256) for layer 1 (32, 512)layer 2 (32, 1024,) layer 3 layer 4 (32, 2048,)
            vectors[name][path] = activations[name][i].mean(dim=[1, 2]).cpu().numpy()
        vector_pids[path]=pids[i].item() # tensor access to plain int
    


print("\n=== Linear Probe Results ===")


for layer in  ['layer1', 'layer2', 'layer3', 'layer4']:
    paths_list=list(vectors[layer].keys()) # gets all the paths that anchor everything
    X=np.stack([vectors[layer][path] for path in paths_list])
    y_color=np.stack([color_labels[path] for path in paths_list])
    y_pid=np.stack([vector_pids[path]for path in paths_list])

    # color probe data split
    X_train,X_test,y_train,y_test=train_test_split(X, y_color, test_size=0.2, random_state=42)
    # instantiate model
    clf=LogisticRegression(max_iter=1000)
    # train 
    clf.fit( X_train,y_train)
    # pred on unseen data
    y_pred=clf.predict(X_test)
    #how accurate
    color_accuracy = accuracy_score(y_test, y_pred)

    print(f"{layer} color probe: {color_accuracy:.3f}")

    # pid prob data split
    X_train,X_test,y_train,y_test=train_test_split(X, y_pid, test_size=0.2, random_state=42)
    # instantiate model
    clf=LogisticRegression(max_iter=1000)
    # train 
    clf.fit( X_train,y_train)
    # pred on unseen data
    y_pred=clf.predict(X_test)
    #how accurate
    pid_accuracy = accuracy_score(y_test, y_pred)

    print(f"{layer} pid probe: {pid_accuracy:.3f}")

    print(f"{layer} | color: {color_accuracy:.3f} | identity: {pid_accuracy:.3f}")



    

