# imports — 
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
import cv2
"""
This script answers the question What does a filter inside a layer of the model(CNN) trained on uncovering?

We load the model with the trained weights and set the model to evaluation mode.
In eval model the model, we do not update the weights they are frozen
Instead we need gradient ascent on the random noise input


"""


model = build_model(
   name=config.MODEL,
   num_classes=751,
   loss='triplet', # outputs (1,751)-class cores, features = self.model(imgs)
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


"""
This function takes in the model.layer, channel idx and in the forward pass 
Once the model gets to a specified layer,register hook runs hook fn and the activations from that layer are captured
"""
# activation maximization function

def maximize_channel(model, layer_name, channel_idx, num_steps=200, lr=0.1):
	captured = {}

	def hook_fn(module, input, output): #incoming feature map is  multiplied by Filter 214's weights to generate Filter 214's output activation values.
		captured['activation'] = output # activation tesnor (batch,2048,7,7)

	# for given model layer whenever model(input_img)runs, and gets to that layer hook_fn is called
	hook = getattr(model, layer_name).register_forward_hook(hook_fn)

    # start from random noise — shape (1, 3, 256, 128) requires_grad=True on the image
	input_img = torch.randn(1, 3, 256, 128).to(device).requires_grad_(True)
	#optimizer
	optimizer = torch.optim.Adam([input_img], lr=lr)
    # forward pass up to target layer
	for step in range(num_steps):
		# clear any gradients
		optimizer.zero_grad()
		# forward pass runs
		output=model(input_img) # model runs towards producing an output

		# score = mean of target channel across spatial dimensions
		score = captured['activation'][0, channel_idx].mean()
		
		loss = -score  # to turn the optimizer that is ideally a minimizer to maximizer reducing -score means maxing score
		loss.backward() # gradients are computed on -score
		optimizer.step()

	hook.remove()  # clean up hook after done
	return input_img.detach()

# sample channels to visualize
layers_and_channels = {
    'layer1': [0, 50, 100, 150, 200],
    'layer2': [0, 50, 100, 150, 200],
    'layer3': [0, 50, 100, 150, 200],
    'layer4': [0, 50, 100, 150, 200],
}
for layer,channels in layers_and_channels.items():
	for channel in channels:
		result_img = maximize_channel(model, layer, channel) 
		# tensor (1, 3, 256, 128) → numpy (256, 128, 3)
		img = result_img.squeeze(0) # remove batch dim → (3, 256, 128)
		img = img.permute(1, 2, 0).cpu().numpy()  # → (256, 128, 3)
		img = (img - img.min()) / (img.max() - img.min()) * 255  # normalize to 0-255
		img = img.astype(np.uint8)   # convert to uint8

		filename = f"{config.OUTPUT_DIR}/actmax/{layer}_channel{channel}.png"
		os.makedirs(f"{config.OUTPUT_DIR}/actmax", exist_ok=True)
		cv2.imwrite(filename, img)
	



