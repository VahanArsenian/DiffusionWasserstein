import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as F
import numpy as np
import os 
import pickle

class FIDScore:

    def __init__(self, device=0):
        self.device = device
        self.fid = FrechetInceptionDistance(normalize=False).to(0)

    @classmethod
    def preprocess_image(cls, image):
        image = torch.tensor(image, dtype=torch.uint8).unsqueeze(0)
        image = image.permute(0, 3, 1, 2)
        return image

    def update(self, image_set_left, image_set_right):
        self.fid.update(image_set_left, real=True)
        self.fid.update(image_set_right, real=False)
        
    def compute(self):
        res = (self.fid.compute())
        print("Cache reset initiated")
        self.fid.reset()
        return res

    def load_and_process_files(self, file_path_left, file_path_right):
        image_left = np.load(file_path_left)
        image_right = np.load(file_path_right)
        if 38*2**30 <= torch.cuda.get_device_properties(0).total_memory <= 40*2**30:
            batch_size = 2048

        print("Comapring # of images: ", len(image_left), len(image_right))
        for i in range(0, len(image_left), batch_size):
            print("Updating FID for images: ", i, i+batch_size)
            image_set_left = torch.cat([self.preprocess_image(image) for image in list(image_left.values())][i:i+batch_size]).to(0)
            image_set_right = torch.cat([self.preprocess_image(image) for image in list(image_right.values())][i:i+batch_size]).to(0)
            print(image_set_left.shape, image_set_right.shape)  
            self.update(image_set_left, image_set_right)



if __name__ == "__main__":
    fid = FIDScore()
    results = {}
    root = "src/data/noisy_score_images/"
    files = [f for f in os.listdir(root) if os.path.isfile(root+f)]
    print(files)
    
    for file in files:
        std = float(file.replace("generated_images_", "").replace(".npz", ""))
        fid.load_and_process_files(root + file, "src/data/cifar10_sample/generated_images.npz")
        fid_to_cifar = fid.compute().item()
        results[std] = fid_to_cifar
        print(f"Std {std} with: FID {fid_to_cifar}")
    
    fid.load_and_process_files("src/data/standard_score_images/generated_images.npz", "src/data/cifar10_sample/generated_images.npz")
    results[0] = fid.compute().item()
    print(f"Pure score FID  is: {results[0]}")
    
    with open('fid_res.pkl', 'wb') as handle:
        pickle.dump(results, handle)
    
