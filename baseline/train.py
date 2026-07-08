import config
import os
import sys

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