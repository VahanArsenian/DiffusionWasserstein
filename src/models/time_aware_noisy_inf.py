import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from patched_ddpm import create_patched_from_pretrained
from argparse import ArgumentParser
from noise import NoiseBuilder  


def run_inference(rank, world_size, model_name, 
                  batch_per_device, result_queue, 
                  noise_dist, std):
    model_id = model_name
    batch_size = batch_per_device
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    g_cuda = torch.Generator(device=rank).manual_seed(42*rank)
    noise = NoiseBuilder.build(noise_dist, std=std, generator=g_cuda)
    
    ddpm = create_patched_from_pretrained(model_id, noise, result_queue).to(rank)

    with torch.no_grad():
        ddpm(batch_size=batch_size, generator=g_cuda, num_inference_steps=1000)



def main(result_queue, n_gpus=8, model_name="google/ddpm-cifar10-32", 
         batch_per_device=4, total_cycles_per_device=1, std=1.0, noise_dist="normal"):
    world_size = n_gpus
    for rank in range(n_gpus):
        mp.Process(target=run_inference, args=(rank, world_size, model_name, batch_per_device, 
                                               total_cycles_per_device, result_queue, noise_dist, std)).start()

def get_results(result_queue):
    results = defaultdict(list)
    while result_queue.empty() is False:
        tmp = result_queue.get()
        results[tmp[0]].append(tmp[1])
    return results


def model_resolver(dataset: str):
    if dataset == "cifar10":
        return "google/ddpm-cifar10-32"
    elif dataset == "celebahq":
        return "google/ddpm-celebahq-256"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--run-path", type=str, default="run_pathscore", help="Path to run_pathscore")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for inference")
    parser.add_argument("--std", type=float, default=1.0, help="std for the distribution")
    parser.add_argument("--noise_dist", type=str, default="normal", help="Noise distribution for the distribution")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use")
    parser.add_argument("--total_cycles_per_device", type=int, default=1, help="Total cycles per device. The total number of images will be total_cycles_per_device * batch_size * n_gpus")

    
    args = parser.parse_args()
    batch_size = args.batch_size
    std = args.std
    noise_dist = args.noise_dist
    dataset = args.dataset
    total_cycles_per_device = args.total_cycles_per_device
    model_name = model_resolver(dataset)


    print(f"Using {noise_dist} noise distribution")
    print(f"Noisy score with std: {std}")
    file_name = "noisy_score.pdf"
    root_folder = f"src/data/temporal/noisy_score_images_{dataset}/"
    store_folder = os.path.join(root_folder, noise_dist)
    os.makedirs(store_folder, exist_ok=True)
    results = defaultdict(list)
    n_gpu = torch.cuda.device_count()
    for i in range(total_cycles_per_device):
        print("Cycle ", i)
        print("===================================")
        result_queue = mp.Queue()
        main(result_queue=result_queue, n_gpus=n_gpu,
            model_name=model_name, batch_per_device=batch_size,
            std=std, noise_dist=noise_dist, total_cycles_per_device=total_cycles_per_device)
        tmp_res = get_results(result_queue=result_queue)
        for k, v in tmp_res.items():
            results[k].append(v)
        result_queue.close()
    for k, v in results.items():
        print("====== Storing results ======")
        print("Storing to file")
        # Save the images to a folder
        generated_as_np = [res.convert("RGB") for res in v]
        np.savez_compressed(f"{store_folder}/{k}/generated_images_{std}.npz", *generated_as_np)
    
    print(f"Results saved in {store_folder}")
    print("===================================")

