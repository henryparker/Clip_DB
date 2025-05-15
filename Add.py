from config import *
from utils import import_embeddings_to_chroma
import torchvision.transforms as transforms
import torchvision
import torch
import chromadb
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from Modules import ImageFolderDataset

# Model
model = CLIPModel.from_pretrained(ClipModel).to(device)
processor = CLIPProcessor.from_pretrained(ClipModel)


# Load Data
transform = transforms.Compose([
    transforms.ToTensor()
])

if IS_CUSTOM_IMAGE:
    imageset = ImageFolderDataset(IMAGE_PATH, transform=transform)
else:
    if IMAGESET_NAME == 'MNIST':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        imageset = torchvision.datasets.MNIST(root=DATA_PATH, transform=transform, train = True, download = True)
    elif IMAGESET_NAME == 'CIFAR10':
        imageset = torchvision.datasets.CIFAR10(root=DATA_PATH, transform=transform, train = True, download = True)
    elif IMAGESET_NAME == 'CIFAR100':
        imageset = torchvision.datasets.CIFAR100(root=DATA_PATH, train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset '{IMAGESET_NAME}' is not supported. Please use 'MNIST', 'CIFAR10', or 'CIFAR100'.")

imageloader = torch.utils.data.DataLoader(imageset, batch_size=batch_size, shuffle=False, num_workers=2)


# Set Chroma Database
client = chromadb.PersistentClient(path=CHROMA_PATH)
# client.delete_collection(name=COLLECTION_NAME)
collection = client.get_or_create_collection(name=COLLECTION_NAME,
                                             metadata={"hnsw:space": "cosine"})

print(f"ChromaDB Collection '{COLLECTION_NAME}' ready.")


# Store Image Vectors into DB
try:
    import_embeddings_to_chroma(imageloader, processor, model, device, collection)
except Exception as e:
    print(f"An error occurred during embedding import: {e}")