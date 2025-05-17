import numpy as np


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

