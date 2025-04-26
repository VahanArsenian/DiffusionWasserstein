import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np


def run_inference(rank, world_size, model_name, batch_per_device, result_queue):
    from diffusers import DDPMPipeline
    model_id = model_name
    batch_size = batch_per_device
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    with torch.no_grad():
        ddpm = DDPMPipeline.from_pretrained(model_id).to(rank)
        g_cuda = torch.Generator(device=rank).manual_seed(42*rank)
        output = ddpm(batch_size=batch_size, generator=g_cuda)
    result_queue.put(output.images)



def main(result_queue, n_gpus=8, model_name="google/ddpm-cifar10-32", 
         batch_per_device=4):
    world_size = n_gpus
    # ctx = mp.get_context('spawn')
    for rank in range(n_gpus):
        mp.Process(target=run_inference, args=(rank, world_size, model_name, batch_per_device, result_queue)).start()

def get_results(n_gpus, result_queue):
    results = []
    for i in range(n_gpus):
        results += (result_queue.get())
    return results


def plot_images_grid(images, cols: int, save_path: str = "test.pdf", base_figsize_per_subplot: tuple[float, float] = (3.0, 3.0)) -> None:
    n_images = len(images)
    rows = np.ceil(n_images / cols).astype(int)

    # Calculate dynamic figure size
    fig_width = cols * base_figsize_per_subplot[0]
    fig_height = rows * base_figsize_per_subplot[1]
    figsize = (fig_width, fig_height)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Flatten axes array for easy iteration, handle single row/col cases
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else: # Handle case with only one subplot (plt.subplots returns a single Axes object)
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
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory


if __name__=="__main__":
    n_gpu = torch.cuda.device_count()
    result_queue = mp.Queue()
    main(result_queue=result_queue, n_gpus=n_gpu,
         model_name="google/ddpm-cifar10-32", batch_per_device=4)
    results = get_results(n_gpus=n_gpu, result_queue=result_queue)
    print("====== Results ===")
    result_queue.close()
    plot_images_grid(results, cols=n_gpu)




