import os
from PIL import Image
from torch.utils.data import Dataset


class TinyImageNetValDataset(Dataset):
    def __init__(self, val_images_path, val_annotations_path, wnid_to_label, transform=None):
        self.val_images_path = val_images_path
        self.val_annotations_path = val_annotations_path
        self.wnid_to_label = wnid_to_label
        self.transform = transform
        self.samples = []

        if not os.path.exists(val_annotations_path):
             raise FileNotFoundError(f"Validation annotations file not found at {val_annotations_path}")
        if not os.path.exists(val_images_path):
             raise FileNotFoundError(f"Validation images directory not found at {val_images_path}")
        if not wnid_to_label:
             raise ValueError("WNID to label mapping is empty or None. Ensure wnid_to_label is created successfully.")

        try:
            with open(val_annotations_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    filename = parts[0]
                    wnid = parts[1]

                    if wnid in self.wnid_to_label:
                         self.samples.append((filename, wnid))
                    else:
                         print(f"Warning: WNID '{wnid}' from {filename} not found in wnid_to_label mapping. Skipping sample.")

        except Exception as e:
            raise IOError(f"Error parsing val_annotations.txt: {e}") from e

        print(f"Parsed {len(self.samples)} samples from {val_annotations_path}.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename, wnid = self.samples[index]

        image_path = os.path.join(self.val_images_path, filename)

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.wnid_to_label[wnid]

        return image, label
    
class ImageFolderDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = []

        if not os.path.isdir(image_folder):
            raise FileNotFoundError(f"Error: File'{image_folder}' doesn't exist or is not a available path.")

        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                self.image_files.append(filename)

        if not self.image_files:
            print(f"Warning: '{image_folder}' has no jpg or jpeg images")

        print(f"Successfully found {len(self.image_files)} images")


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_filename)

        label = os.path.splitext(img_filename)[0]

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"错误: 无法加载图片 {img_path} - {e}")
            return None, None

        if self.transform:
            img = self.transform(img)

        return img, label