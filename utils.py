import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
import chromadb
from chromadb.api.models.Collection import Collection
from typing import Union
import cv2
import os

def import_embeddings_to_chroma(
    imageloader: DataLoader,
    processor: CLIPProcessor, # Replace with the actual processor type if needed
    model: CLIPModel,       # Replace with the actual model type if needed
    device: Union[str, torch.device],
    collection: Collection,
    description: str = "Importing Embeddings"
) -> None:
    for i, (images, labels) in enumerate(tqdm(imageloader, desc=description)):
        with torch.no_grad():
            clip_inputs = processor(images=images, return_tensors="pt", padding=True)
            pixel_values = clip_inputs.pixel_values.to(device)

            image_features = model.get_image_features(pixel_values=pixel_values)

            batch_embeddings = image_features.cpu().numpy().tolist()

            batch_ids = []
            batch_metadatas = []

            for j in range(images.size(0)):
                unique_id = f"{i * imageloader.batch_size + j}"
                batch_ids.append(unique_id)

                metadata_item = {
                    "label": labels[j]
                }
                batch_metadatas.append(metadata_item)

            collection.add(
                    embeddings=batch_embeddings,
                    ids=batch_ids,
                    metadatas=batch_metadatas,
                )
    print("Embedding Import completed.")

def extract_frames_every_x_seconds(video_path, output_folder='Imagefolder', interval_seconds=5, name='img'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Can't open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"Property:")
    print(f"FPS: {fps}")
    print(f"Frames Count: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")

    frame_interval = int(fps * interval_seconds)
    if frame_interval <= 0:
        frame_interval = 1

    saved_frame_count = 0
    current_frame_index = 0

    while current_frame_index < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)

        ret, frame = cap.read()

        if ret:
            time_in_seconds = current_frame_index / fps
            filename = os.path.join(output_folder, f"{name}_{time_in_seconds:.2f}s.jpg")

            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            saved_frame_count += 1

            current_frame_index += frame_interval
        else:
            break

    cap.release()
    print(f"Frame extraction is Done, saving {saved_frame_count} frames into {output_folder}")