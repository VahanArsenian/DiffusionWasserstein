import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from patched_ddpm import TemporalDDPMPipeline
from argparse import ArgumentParser
from noise import NoiseBuilder  


mp.set_start_method("spawn", force=True)



def run_inference(rank: int, batch_size: int, model_name: str, 
                  noise_dist: str, std: float, timesteps: set[int]) -> dict[str, list[np.ndarray]]:
    # seed & build noise
    g_cuda = torch.Generator(device=rank).manual_seed(42*rank)
    noise = NoiseBuilder.build(noise_dist, std=std, generator=g_cuda)

    ddpm = TemporalDDPMPipeline.from_pretrained(model_name, noise, timesteps=timesteps).to(rank)

    with torch.no_grad():
        return ddpm(batch_size=batch_size, generator=g_cuda, num_inference_steps=1000)

def gather_results(local: dict[str, list[np.ndarray]]) -> dict[str, list[np.ndarray]]:
    """Gather Python objects from all ranks into a single dict on rank 0."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank == 0:
        # only destination rank provides a gather_list
        gathered: list = [None] * world_size  # type: ignore
        dist.gather_object(local, gathered, dst=0)
        # merge results
        merged: dict[str, list[np.ndarray]] = defaultdict(list)
        for d in gathered:
            for k, v in d.items():
                merged[k] += v
        return merged
    else:
        # non-destination ranks pass None
        dist.gather_object(local, None, dst=0)
        return {}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run-path", type=str, default="run_pathscore", help="Path to run_pathscore")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for inference")
    parser.add_argument("--std", type=float, default=1.0, help="std for the distribution")
    parser.add_argument("--noise_dist", type=str, default="normal", help="Noise distribution for the distribution")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use")
    parser.add_argument("--total_cycles_per_device", type=int, default=1, help="Total cycles per device. The total number of images will be total_cycles_per_device * batch_size * n_gpus")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="Number of processes to run per node")
    parser.add_argument("--root_folder_prefix", type=str, default="src/data/temporal/noisy_score_images", help="Root folder to store images")
    parser.add_argument("--timesteps", type=int, default=0, nargs="+", help="Timesteps to return")
    args = parser.parse_args()

    dataset_to_model = {
        "cifar10": "google/ddpm-cifar10-32",
        "celebahq": "google/ddpm-celebahq-256"
    }

    # torchrun sets these ENV vars
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # bind this process to its GPU and initialize NCCL
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method="env://")

    batch_size = args.batch_size
    std = args.std
    noise_dist = args.noise_dist
    dataset = args.dataset
    total_cycles_per_device = args.total_cycles_per_device
    timesteps = args.timesteps

    root_folder = f"{args.root_folder_prefix}_{dataset}/"
    store_folder = os.path.join(root_folder, noise_dist)
    model_name = dataset_to_model[dataset]

    print(world_size, rank)
    print(f"Reading timesteps: {timesteps}")
    print(f"Using {noise_dist} noise distribution")
    print(f"Noisy score with std: {std}")
    os.makedirs(store_folder, exist_ok=True)
    results = defaultdict(list)

    for i in range(total_cycles_per_device):
        print("Cycle ", i)
        print("===================================")
        local = run_inference(rank, batch_size, model_name, noise_dist, std, timesteps)

        # synchronize before gathering
        dist.barrier()
        # all ranks participate in gather to avoid deadlock 
        merged = gather_results(local)
        if rank == 0:
            for k, v in merged.items():
                results[k] += v

    # destroy NCCL process group after all cycles
    dist.destroy_process_group()
    if rank == 0:
        print(results.keys())
        for k, v in results.items():
            print("====== Storing results ======", k, len(v))
            print("Storing to file")
            # Save the images to a folder
            # generated_as_np = [res.convert("RGB") for res in v]
            os.makedirs(f"{store_folder}/{k}", exist_ok=True)
            np.savez_compressed(f"{store_folder}/{k}/generated_images_{std}.npz", *v)
    
        print(f"Results saved in {store_folder}")
        print("===================================")



# torchrun --standalone src/models/time_aware_noisy_inf.py --run-path ./ --batch_size=1024 --std=1 --noise_dist="normal" --dataset=cifar10

