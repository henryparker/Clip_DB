from config import *
import torchvision.transforms as transforms
import torchvision
import torch
import chromadb
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import os
from Modules import TinyImageNetValDataset, ImageFolderDataset

model = CLIPModel.from_pretrained(ClipModel).to(device)
processor = CLIPProcessor.from_pretrained(ClipModel)

data_path = './data'

transform = transforms.Compose([
    transforms.ToTensor()
])

imageset = ImageFolderDataset(SEARCH_IMAGE_PATH, transform=transform)
imageloader = torch.utils.data.DataLoader(imageset, batch_size=1, shuffle=False, num_workers=2)

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_collection(name=COLLECTION_NAME)
print(f"ChromaDB Collection '{COLLECTION_NAME}' ready.")

# top1acccount = 0
# top5acccount = 0
for image, label in tqdm(imageloader):
    with torch.no_grad():
        clip_inputs = processor(images=image, return_tensors="pt", padding=True)
    
        query_embedding = model.get_image_features(pixel_values=clip_inputs.pixel_values.to(device))
        query_embedding_list = query_embedding.cpu().numpy().tolist()
        results = collection.query(
                query_embeddings=query_embedding_list,
                n_results=5,
                include=['metadatas', 'distances']
            )
        print(f'Image {label} has similar images in dataset:')
        print(results)
        # # Top 1
        # if results['metadatas'][0][0]['label'] == label.item():
        #     top1acccount += 1
        # # Top 5
        # ll = set([i['label'] for i in results['metadatas'][0]])
        # if label.item() in ll:
        #     top5acccount += 1

# print(f"Top1 Accuracy:{top1acccount / len(testset) * 100}%")
# print(f"Top5 Accuracy:{top5acccount / len(testset) * 100}%")
