'''The stress test is essentially asking: 
"does this model track identity or does it track shirts?"

If the model passes — it finds the correct person despite the shirt color change.
Identity features (body shape, proportions, gait patterns) are strong enough to overcome the color mismatch. 
The model is robust.

If the model fails — it returns people wearing similar colored shirts to the original, not the correct person.
The color shortcut is so dominant that changing the shirt color breaks the match entirely.
That failure rate is your severity score.'''

import cv2

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
		img_path, pid, camid = self.query_data[index]
		# load image
		orig_img = cv2.imread(img_path)
		if orig_img is None:
			raise FileNotFoundError(f"Image not found at: {img_path}")
		# crop org image
		img_torso = orig_img[70:170, :]
		# convert to HSV
		img_hsv = cv2.cvtColor(img_torso, cv2.COLOR_BGR2HSV)

