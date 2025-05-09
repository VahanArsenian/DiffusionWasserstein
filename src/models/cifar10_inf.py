import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from patched_ddpm import create_patched_from_pretrained
from argparse import ArgumentParser
from noise import NoiseBuilder  


def preset_params(ddpm, noise):
    ddpm.scheduler.noise_gen = noise

def run_inference(rank, world_size, model_name, 
                  batch_per_device, result_queue, 
                  noise_dist, std):
    model_id = model_name
    batch_size = batch_per_device
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    g_cuda = torch.Generator(device=rank).manual_seed(42*rank)
    noise = NoiseBuilder.build(noise_dist, std=std, generator=g_cuda)

    with torch.no_grad():
        ddpm = create_patched_from_pretrained(model_id).to(rank)
        preset_params(ddpm, noise)
        output = ddpm(batch_size=batch_size, generator=g_cuda, num_inference_steps=1000)
    result_queue.put(output.images)



def main(result_queue, n_gpus=8, model_name="google/ddpm-cifar10-32", 
         batch_per_device=4, std=1.0, noise_dist="normal"):
    world_size = n_gpus
    for rank in range(n_gpus):
        mp.Process(target=run_inference, args=(rank, world_size, model_name, batch_per_device, 
                                               result_queue, noise_dist, std)).start()

def get_results(n_gpus, result_queue):
    results = []
    for i in range(n_gpus):
        results += (result_queue.get())
    return results


def model_resolver(dataset: str):
    if dataset == "cifar10":
        return "google/ddpm-cifar10-32"
    elif dataset == "celebahq":
        return "google/ddpm-celebahq-256"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def plot_images_grid(images, cols: int, save_path: str = "test.pdf", base_figsize_per_subplot: tuple[float, float] = (3.0, 3.0)) -> None:
    n_images = len(images)
    rows = np.ceil(n_images / cols).astype(int)


    fig_width = cols * base_figsize_per_subplot[0]
    fig_height = rows * base_figsize_per_subplot[1]
    figsize = (fig_width, fig_height)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else: 
        axes = [axes]


    for i, ax in enumerate(axes):
        if i < n_images:
            try:
                ax.imshow(images[i])
                ax.set_title(f"Image {i+1}") # Optional: Add title
            except Exception as e:
                ax.text(0.5, 0.5, 'Error loading image', horizontalalignment='center', verticalalignment='center')
            ax.axis("off")  # Hide axes ticks and labels
        else:
            ax.axis("off")  # Hide unused subplots

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory
    else:
        plt.show()


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--store_to_file", action="store_true", help="Should store to a folder")
    parser.add_argument("--run-path", type=str, default="run_pathscore", help="Path to run_pathscore")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for inference")
    parser.add_argument("--std", type=float, default=1.0, help="std for the distribution")
    parser.add_argument("--noise_dist", type=str, default="normal", help="Noise distribution for the distribution")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use")

    
    args = parser.parse_args()
    store_to_file = args.store_to_file
    batch_size = args.batch_size
    std = args.std
    noise_dist = args.noise_dist
    dataset = args.dataset
    model_name = model_resolver(dataset)

    print(f"Using {noise_dist} noise distribution")
    print(f"Noisy score with std: {std}")
    file_name = "noisy_score.pdf"
    root_folder = f"src/data/noisy_score_images_{dataset}/"
    store_folder = os.path.join(root_folder, noise_dist)
    os.makedirs(store_folder, exist_ok=True)

    n_gpu = torch.cuda.device_count()
    result_queue = mp.Queue()
    main(result_queue=result_queue, n_gpus=n_gpu,
         model_name=model_name, batch_per_device=batch_size,
         std=std, noise_dist=noise_dist)
    results = get_results(n_gpus=n_gpu, result_queue=result_queue)
    print("====== Results saved to file ======")
    
    result_queue.close()
    if store_to_file:
        print("Storing to file")
        # Save the images to a folder
        generated_as_np = [res.convert("RGB") for res in results]
        np.savez_compressed(f"{store_folder}/generated_images_{std}.npz", *generated_as_np)
    else:
        print("Storing to pdf")
        # Save the images to a single PDF file
        plot_images_grid(results, cols=n_gpu, save_path=file_name)
    print(f"Results saved")
    print("===================================")




