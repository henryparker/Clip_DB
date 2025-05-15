import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
ClipModel = "openai/clip-vit-large-patch14"
batch_size = 64

DATA_PATH = './data'

CHROMA_PATH = './client'
COLLECTION_NAME = 'TomJerry'

## Adding images into database
# Select:
#   1. custom image folder OR
#   2. common image set
IS_CUSTOM_IMAGE = True

# 1. Path of custom images adding to database
IMAGE_PATH = './data/TomJerry'

# 2. Common image set
IMAGESET_NAME = 'MNIST' # {'MNIST', 'CIFAR10', 'CIFAR100'}

## Search Similar images in database
SEARCH_IMAGE_PATH = './data/TomJerryTest'
