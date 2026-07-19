# This file helps you set up in you own enviroment 
#Don't push your config.py to github
#Copy and paste this code in config.py to help you run the program on colab or on your local machine

DATASET_ROOT="" # eg./root/.cache/kagglehub/datasets/pengcw1/market-1501/versions/1(colab)
                # eg./Users/yourname/projects/reid-shortcut-audit/data (local)
OUTPUT_DIR=""   # save the progress made during training either in outputs file on collab or local machine
BATCH_SIZE= 32  #reduce if you run out of GPU memory
DEVICE= ""    #'cuda' on Colab, 'cpu' or 'mps' on Mac
LR = 0.0003
EPOCH= 60
MODEL='resnet50'