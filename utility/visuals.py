
import matplotlib.pyplot as plt  # For plotting visualizations
from tabulate import tabulate  # For formatted table printing
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from collections import Counter
from scipy.ndimage import gaussian_filter

def plot_output(
        output,
        x,
        y,
        filter=False,
        saving_path=f'../fig/gan_output/gassian_plot.png'
    ):
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axes_list = plt.subplots(len(output), 3, figsize=(10, 100))

    # Gaussian filter parameters (adjust these for your case)
    sigma = 1.2  # Standard deviation for the Gaussian filter

    for i in range(len(output)):
        axes = axes_list[i].flatten()

        # Apply Gaussian filter to spread out pixel values
        output_spread = gaussian_filter(output[i], sigma=sigma) \
            if filter else output[i]

        axes[0].imshow(gaussian_filter(x[i][0], sigma=sigma))
        axes[1].imshow(gaussian_filter(y[i][0] + x[i][0], sigma=sigma))
        axes[2].imshow(output_spread)

        if i == 0:
            axes[0].set_title('Input Itinerary', fontweight='bold')
            axes[1].set_title('Expected Itinerary', fontweight='bold')
            axes[2].set_title('Generated Itinerary (Spread)', fontweight='bold')

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

    # Save the figure with spread-out images
    plt.savefig(saving_path)


# =============================================================================
# Function: print_table
# =============================================================================
def print_table(headers, table, prefix=None):
    """
    Print a formatted table using the tabulate library.
    
    Parameters:
    - headers (list): List of header names.
    - table (list of lists): Data rows for the table.
    - prefix (str, optional): Text to print before the table.
    """
    # Print the prefix if provided.
    if prefix is not None:
        print(prefix)
    # Print the table in a formatted style.
    print(tabulate(table, headers, tablefmt='fancy_grid', stralign='center'))


def visualize_matrix(matrix, title='Matrix Covering Float Coordinates with Min and Max',
                     x_label='X Coordinate', y_label='Y Coordinate'):
    """
    Visualize a matrix as a heatmap.
    
    Parameters:
    - matrix (ndarray): Matrix to be visualized.
    - title (str): Plot title.
    - x_label (str): Label for the x-axis.
    - y_label (str): Label for the y-axis.
    """
    plt.imshow(matrix, origin='lower', cmap='Greys')
    plt.colorbar(label='Matrix Value')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def evaluate_and_plot(test_loader, model, encoder, title, dataset_name):
    def jaccard_sim(pred, truth):
        pred_set = set(pred)
        truth_set = set(truth)
        return len(pred_set & truth_set) / len(pred_set | truth_set)
    
    # Set plotting style
    plt.rcParams["font.family"] = "Times New Roman"
    
    jc_list = []
    emd_list = []
    total_number_of_samples = len(test_loader.dataset)
    cols = 5
    rows = (total_number_of_samples // cols) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    
    counter = Counter()
    
    # Loop over the dataset samples
    for i, (test_x, test_y) in enumerate(test_loader.dataset):
        ax = axes[i]
        
        # Prepare the input image
        test_x = test_x.to(model.device)
        test_x = test_x.unsqueeze(0)
        
        # Get indices where test_y is nonzero.
        # [0] selects the first set of indices if multiple dimensions exist.
        truth = np.argwhere(test_y != 0)[0].detach().numpy()
    
        # Run inference
        raw_pred = model.inference(test_x)[0]
        pred = torch.argwhere(raw_pred > 0.5).squeeze(1)
        if len(pred) == 0:
            pred = torch.argmax(raw_pred).unsqueeze(0)
            print('miss')
        pred = pred.cpu().detach().numpy()
    
        jc_score = 1 - jaccard_sim(pred, truth)
    
        # Transform predictions and ground truth using the encoder
        pred = encoder.transform(pred, inverse=True)
        truth = encoder.transform(truth, inverse=True)
    
        truth_np = np.array(truth)
        pred_np = np.array(pred)
        counter.update([tuple(ele) for ele in pred_np])
    
        ax.scatter(truth_np[:, 0], truth_np[:, 1], c='blue', s=30,
                   label='ground truth' if i == 0 else None)
        ax.scatter(pred_np[:, 0], pred_np[:, 1], c='red', s=10,
                   label='prediction' if i == 0 else None)
    
        # Compute Earth Mover's Distance using separate dimensions
        emd_x = wasserstein_distance(pred_np[:, 0], truth_np[:, 0])
        emd_y = wasserstein_distance(pred_np[:, 1], truth_np[:, 1])
        emd = (emd_x + emd_y) / 2
    
        ax.set_title(f"Sample {i+1}, emd {emd:.2f}, jc {jc_score:.2f}",
                     fontsize=12, fontweight='bold')
        jc_list.append(jc_score)
        emd_list.append(emd)
    
    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    if total_number_of_samples > 0:
        axes[0].legend(loc='upper left', bbox_to_anchor=(-.6, 1))
    
    fig.suptitle(f"Model {title}'s Performance on Dataset {dataset_name}\nAverage EMD {np.mean(emd_list):.2f}\nAverage Jaccard Score {np.mean(jc_list):.2f}", fontweight='bold', fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.legend()
    plt.show()
    return fig, jc_list, emd_list, counter