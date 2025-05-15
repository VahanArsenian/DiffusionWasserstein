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


mp.set_start_method("spawn", force=True)



def run_inference(rank: int, world_size: int, batch_size: int,
                  model_name: str, noise_dist: str, std: float) -> dict[str, list[np.ndarray]]:
    # seed & build noise
    g_cuda = torch.Generator(device=rank).manual_seed(42*rank)
    noise = NoiseBuilder.build(noise_dist, std=std, generator=g_cuda)
    
    ddpm = create_patched_from_pretrained(model_name, noise, temporal=True).to(rank)

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


def model_resolver(dataset: str):
    if dataset == "cifar10":
        return "google/ddpm-cifar10-32"
    elif dataset == "celebahq":
        return "google/ddpm-celebahq-256"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run-path", type=str, default="run_pathscore", help="Path to run_pathscore")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for inference")
    parser.add_argument("--std", type=float, default=1.0, help="std for the distribution")
    parser.add_argument("--noise_dist", type=str, default="normal", help="Noise distribution for the distribution")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use")
    parser.add_argument("--total_cycles_per_device", type=int, default=1, help="Total cycles per device. The total number of images will be total_cycles_per_device * batch_size * n_gpus")
    parser.add_argument("--nproc-per-node", type=int, default=8, help="Path to run_pathscore")
    
    args = parser.parse_args()

    # torchrun sets these ENV vars
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # bind this process to its GPU and initialize NCCL
    import torch
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method="env://")

    model_map = {"cifar10": "google/ddpm-cifar10-32",
                  "celebahq": "google/ddpm-celebahq-256"}
    batch_size = args.batch_size
    std = args.std
    noise_dist = args.noise_dist
    dataset = args.dataset
    total_cycles_per_device = args.total_cycles_per_device
    model_name = model_resolver(dataset)

    print(world_size, rank)
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
        local = run_inference(rank, world_size, batch_size, model_name, noise_dist, std)
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

