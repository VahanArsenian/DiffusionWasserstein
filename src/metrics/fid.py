import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as F
import numpy as np
import os 
import pickle
from argparse import ArgumentParser



class FIDScore:

    def __init__(self, device=0, batch_size=2048, num_features=2048, reset_real_features=True):
        self.device = device
        # Note: The implementation of reset sucks, we should reimplement it
        self.fid = FrechetInceptionDistance(normalize=False, feature=num_features, reset_real_features=reset_real_features).to(0)
        self.batch_size = batch_size
        print("Init FID with device: ", self.device, num_features, self.batch_size)

    @classmethod
    def preprocess_image(cls, image):
        image = torch.tensor(image, dtype=torch.uint8).unsqueeze(0)
        image = image.permute(0, 3, 1, 2)
        return image

    def compute(self):
        res = (self.fid.compute())
        print("Cache reset initiated")
        self.fid.reset()
        return res

    def load_and_process_file(self, file, real):
        images = np.load(file)
        batch_size = self.batch_size

        print("Comapring # of images: ", len(images))
        for i in range(0, len(images), batch_size):
            print("Updating FID for images: ", i, i+batch_size)
            image_set = torch.cat([self.preprocess_image(image) for image in list(images.values())][i:i+batch_size]).to(0)
            print(image_set.shape)  
            self.fid.update(image_set, real)


def folder_eval(root_gen, path_real, force=False, batch_size=2048, num_features=2048):
    fid = FIDScore(batch_size=batch_size, num_features=num_features, reset_real_features=False)    
    
    print("Computing real features once!")
    
    files = [f for f in os.listdir(root_gen) if os.path.isfile(root_gen+f)]
    if "fid_cache.pkl" in files and not force:
        results = read_cache(root_gen)
    else:
        results = {}

    files = list(filter(lambda x: x != "fid_cache.pkl", files))
    stds = [float(file.replace("generated_images_", "").replace(".npz", "")) for file in files]

    if set(stds) != set(results.keys()) or force:
        fid.load_and_process_file(path_real, real=True)

    print(files)
    for file, std in zip(files, stds):
        if std in results:
            print(f"Std {std} already computed")
            continue
        print(f"Computing for {file}")
        fid.load_and_process_file(root_gen + file, real=False)
        fid_to_cifar = fid.compute().item()
        results[std] = fid_to_cifar
        print(f"Std {std} with: FID {fid_to_cifar}")
    write_cache(root_gen, results)
    

def read_cache(root):
    with open(os.path.join(root, "fid_cache.pkl"), 'rb') as handle:
        return pickle.load(handle)

def write_cache(root, results):
    with open(os.path.join(root, "fid_cache.pkl"), 'wb') as handle:
        pickle.dump(results, handle)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force recompute")
    parser.add_argument("--noisy_folder", type=str, help="Folder of images with noisy score")
    parser.add_argument("--real_images", type=str, help="Folder of real images"
    )
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for FID computation")
    parser.add_argument("--num_features", type=int, default=2048, help="Number of features for FID computation")
    args = parser.parse_args()

    force = args.force
    root_gen = args.noisy_folder
    path_real = args.real_images
    batch_size = args.batch_size
    num_features = args.num_features

    print("Params ", force, root_gen, path_real, batch_size, num_features)

    folder_eval(root_gen, path_real, force, batch_size=batch_size, num_features=num_features)

# python src/metrics/fid.py --noisy_folder=src/data/noisy_score_images_cifar10/normal/ --real_images=src/data/cifar10_sample/real_images.npz 
# python src/metrics/fid.py --noisy_folder=src/data/noisy_score_images_cifar10/uniform/ --real_images=src/data/cifar10_sample/real_images.npz 

# python src/metrics/fid.py --noisy_folder=src/data/noisy_score_images_celebahq/normal/ --real_images=src/data/celebahq_sample/real_images.npz --num_features=64
# python src/metrics/fid.py --noisy_folder=src/data/noisy_score_images_celebahq/uniform/ --real_images=src/data/celebahq_sample/real_images.npz --num_features=64