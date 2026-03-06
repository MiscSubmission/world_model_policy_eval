import mediapy as media
import numpy as np
from PIL import Image

def sample_and_concatenate_gif(gif_path, output_path, num_frames=6):
    """
    Reads a GIF file, sub-samples `num_frames` frames, and concatenates them horizontally.
    Saves the result to `output_path`.
    """
    # Read all frames from the GIF using PIL
    with Image.open(gif_path) as im:
        frames = []
        try:
            while True:
                frame = im.convert("RGBA")
                frames.append(np.array(frame))
                im.seek(im.tell() + 1)
        except EOFError:
            pass

    frames = np.stack(frames, axis=0)  # shape: (num_total_frames, H, W, C)
    total_frames = frames.shape[0]

    if total_frames == 0:
        raise ValueError("No frames found in GIF.")

    # Sub-sample indices
    if total_frames < num_frames:
        indices = list(range(total_frames)) + [total_frames-1] * (num_frames - total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    sampled_frames = frames[indices]  # shape: (num_frames, H, W, C)

    # Concatenate horizontally
    concat_img = np.concatenate(sampled_frames, axis=1)  # along width

    # Save result
    media.write_image(output_path, concat_img)


sample_and_concatenate_gif("push_t_succ_0.gif", "push_t_succ_0.png")
sample_and_concatenate_gif("push_t_succ_1.gif", "push_t_succ_1.png")
sample_and_concatenate_gif("push_t_fail_0.gif", "push_t_fail_0.png")
sample_and_concatenate_gif("push_t_fail_1.gif", "push_t_fail_1.png")


import matplotlib.pyplot as plt

def plot_correlation_graph(x, y, xlabel="X", ylabel="Y", title="Correlation Graph", output_path="correlation_graph.png"):
    """
    Plots a scatter plot to visualize the correlation between x and y.
    """
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.7)
    
    # Plot regression curve
    # Fit a line: y = a*x + b
    coeffs = np.polyfit(x, y, 1)
    reg_x = np.linspace(0,1, 100)
    reg_y = np.polyval(coeffs, reg_x)
    plt.plot(reg_x, reg_y, color='C0', linewidth=2, label="Cosmos-Predict2-2B")
    plt.plot([0, 1], [0, 1], color='grey', linewidth=2, label="Oracle", alpha=0.5)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # plt.show()
    plt.savefig(output_path, dpi=300)

# Example usage:
np.random.seed(42)
x = np.random.rand(10)
y = x + np.random.normal(0, 0.05, 10)
plot_correlation_graph(x, y, xlabel="Actual Success Rate", ylabel="Predicted Success Rate", title="Push-T", output_path="correlation_graph.png")
