# =============================================================================
# Imports and Dependencies
# =============================================================================
import os  # For file and directory operations
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import torch  # For deep learning tensor operations
import pickle  # For serializing objects
from copy import deepcopy  # To create deep copies of objects
from typing import Tuple  # For type hints
from tqdm import tqdm  # For progress bars during iterations



def get_x_y(labels, encoder, MAX_LEN):
    """
    Process input label data to generate several representations:
    
    For each observation in the provided 'labels' data (a collection of NumPy arrays), this function:
    
    - Extracts unique nonzero values (ignoring the zero background).
    - Skips the observation if it contains only one unique nonzero value.
    - Sorts the unique values to establish a consistent order.
    - Determines a midpoint (half) of the unique values to separate condition values from label values.
    - Creates:
        * x: A modified feature matrix where:
            - For condition values (<= half), entries are set to 1.
            - For label values (> half), entries are set to 0.
        * y: A label matrix where condition values are zeroed out and label values remain.
               This matrix is normalized by dividing by the maximum unique value.
        * y_one_hot: A one-hot encoded version of the label matrix where all nonzero entries
                     are initially set to 1 and then condition values are zeroed out.
        * y_seq: A sequence of encoded labels using the provided encoder (via its `single_transform`
                 method). For each unique label value above the midpoint, the first occurrence's
                 coordinate is encoded and appended to the sequence.
    - Pads the label sequence (y_seq) with zeros to ensure it reaches the length MAX_LEN.
    - Flattens y_one_hot into a one-dimensional array.
    
    Parameters:
    - labels (iterable): A collection (e.g., list) of NumPy arrays representing observations.
    - encoder: An instance that implements a `single_transform` method to encode coordinates.
    - MAX_LEN (int): The maximum sequence length for label sequences (used for padding).
    
    Returns:
    - Tuple containing:
        - x_list (list): List of modified feature matrices.
        - y_list (list): List of normalized label matrices.
        - y_one_hot_list (list): List of flattened one-hot encoded label arrays.
        - y_seq_list (list): List of label sequences.
    """
    x_list = []          # List to store modified feature matrices.
    y_list = []          # List to store normalized label matrices.
    y_seq_list = []      # List to store sequences of encoded label values.
    y_one_hot_list = []  # List to store flattened one-hot encoded representations.

    # Iterate over each observation in the provided label data.
    for observation in labels:
        # Extract unique nonzero values (ignoring the zero background).
        unique_values = np.unique(observation)[1:]
        
        # Skip the observation if it contains only one unique nonzero value.
        if len(unique_values) == 1:
            continue
        
        # Sort the unique values to ensure consistent ordering.
        unique_values = np.sort(unique_values)
        # Calculate the midpoint to separate condition values from label values.
        half = len(unique_values) // 2

        # Create deep copies for processing:
        x = deepcopy(observation)      # Feature matrix to be modified.
        y = deepcopy(observation)      # Label matrix that will be normalized.
        y_seq = []                     # List to hold the sequence of encoded labels.
        y_one_hot = deepcopy(observation)  # Initialize one-hot representation.
        # Set all nonzero elements to 1 for the one-hot encoding.
        y_one_hot[np.where(y_one_hot != 0)] = 1

        # Process each unique value in the observation.
        for val in unique_values:
            # Find indices where the observation equals the current unique value.
            index = np.where(observation == val)
            # Encode the coordinate of the first occurrence using the provided encoder.
            coor = encoder.single_transform((index[0][0], index[1][0]))
            if val > half:
                # For label values (values above half):
                # - Set corresponding entries in x to 0.
                # - Append the encoded coordinate to y_seq.
                x[index] = 0
                y_seq.append(coor)
            else:
                # For condition values (values at or below half):
                # - Set corresponding entries in x to 1.
                # - Zero out these entries in y and y_one_hot.
                x[index] = 1
                y_one_hot[index] = 0
                y[index] = 0
        
        # Normalize the label matrix by dividing by the maximum unique value.
        y = y / max(unique_values)
        # Pad the label sequence with zeros to reach the length MAX_LEN.
        y_seq += [0] * (MAX_LEN - len(y_seq))
        # Flatten the one-hot encoded matrix into a one-dimensional array.
        y_one_hot = y_one_hot.flatten()

        # Append the processed outputs to their respective lists.
        x_list.append(x)
        y_list.append(y)
        y_one_hot_list.append(y_one_hot)
        y_seq_list.append(y_seq)
    
    return x_list, y_list, y_one_hot_list, y_seq_list

# =============================================================================
# Function: process_and_transform_data
# =============================================================================
def process_and_transform_data(file, dir='../processed_data', resolution=0.06, overwrite=False):
    """
    Process and transform data from a CSV file, create matrices and labels, and save the results as a pickle file.
    
    Parameters:
    - file (str): Path to the input CSV file.
    - dir (str): Directory to save the processed pickle file.
    - resolution (float): Resolution for discretizing coordinates.
    - overwrite (bool): If True, overwrite existing processed file.
    
    Returns:
    - str: Path to the processed pickle file.
    """
    # Extract a base file name from the file path.
    file_name = file.split("\\")[-1].split("_")[0]
    # Construct the output file path with a .pkl extension.
    path_name = os.path.join(dir, file_name + '.pkl')
    
    # If the file already exists and overwrite is False, return the existing path.
    if os.path.exists(path_name) and not overwrite:
        print(f"File Found: {path_name}.")
        return path_name
    
    # Load CSV data into a DataFrame.
    df = pd.read_csv(file)
    # Convert string representations of coordinates into actual Python objects.
    df['coordinate'] = df['coordinate'].apply(eval)
    
    # Obtain all unique coordinate entries from the 'coordinate' column.
    all_coordinates = get_unique_list_entries(df, 'coordinate')
    
    # Create matrices for each coordinate entry using the provided resolution and global bounds.
    df['matrices'] = df['coordinate'].apply(lambda x: create_matrix(
        np.array(x),
        resolution,
        **get_min_max_from_matrix(all_coordinates)
    ))
    
    # Retrieve the dimensions (width and height) of the matrix from the first entry.
    WIDTH, HEIGHT = df['matrices'].values[0].shape
    
    # Initialize an encoder to transform the matrix into label sequences.
    encoder = ImageLabelEncoder(width=WIDTH, height=HEIGHT)
    vocab_size = encoder.get_vocab_size()  # Get the vocabulary size (all possible encoded values)
    
    # Generate label sequences from the matrices.
    df['labels'] = df['matrices'].apply(encoder.matrix_transform)
    labels = df['matrices'].values.tolist()
    
    print(f'original dataset size: {len(labels)}')
    # Remove duplicate matrices.
    labels = np.unique(labels, axis=0)
    print(f'dataset size with duplicates removed: {len(labels)}')
    
    # Determine the maximum length among all label sequences.
    MAX_LEN = df['labels'].apply(len).max()
    
    # Create a dictionary of processed data and metadata.
    result_dict = {
        "file name": file_name,
        "vocab_size": vocab_size,
        "max length": MAX_LEN,
        "labels": labels,
        "width": WIDTH,
        "height": HEIGHT,
        'encoder': encoder
    }
    
    # Save the dictionary as a pickle file.
    with open(path_name, 'wb') as f:
        pickle.dump(result_dict, f)
    
    return path_name


# =============================================================================
# Function: get_unique_list_entries
# =============================================================================
def get_unique_list_entries(df: pd.DataFrame, col):
    """
    Retrieve a unique list of entries from a DataFrame column that contains list-like items.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - col (str): Column name with list-like elements.
    
    Returns:
    - np.ndarray: Array of unique entries.
    """
    df = df.copy()  # Work with a copy of the DataFrame
    # If the column entries are strings, evaluate them to convert to list/tuple.
    if isinstance(df[col].values[0], str):
        df[col] = df[col].apply(eval)
    # Flatten the lists, convert to a set for uniqueness, and then to an array.
    return np.array(list(set([sub for ele in df[col].values.tolist() for sub in ele])))

# =============================================================================
# Function: get_min_max_from_matrix
# =============================================================================
def get_min_max_from_matrix(matrix: np.ndarray):
    """
    Calculate the minimum and maximum x and y values from a set of coordinates.
    
    Parameters:
    - matrix (np.ndarray): Array of coordinates.
    
    Returns:
    - dict: Dictionary with global minima and maxima for x and y.
    """
    return {
        "global_x_min": matrix[:, 0].min(),
        "global_x_max": matrix[:, 0].max(),
        "global_y_min": matrix[:, 1].min(),
        "global_y_max": matrix[:, 1].max()
    }

# =============================================================================
# Function: batch_predict
# =============================================================================
def batch_predict(model: torch.nn.Module, x, batch_size=32):
    """
    Perform batch predictions using a PyTorch model.
    
    Parameters:
    - model (torch.nn.Module): Model to use for predictions.
    - x (np.ndarray or torch.Tensor): Input data.
    - batch_size (int): Number of samples per batch.
    
    Returns:
    - torch.Tensor: Concatenated predictions.
    """
    # Convert numpy array to torch tensor if needed.
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    
    # Split the tensor into batches along the first dimension.
    batches = x.split(batch_size, dim=0)
    pred_list = []
    
    # Iterate over each batch and perform model prediction.
    for batch in tqdm(batches):
        pred = model(batch)
        pred_list.append(pred)
    
    # Concatenate all predictions along the first dimension.
    return torch.concat(pred_list)


# =============================================================================
# Function: get_data
# =============================================================================
def get_data(data_path):
    """
    Load all .npy files from the specified directory into a dictionary.
    
    Parameters:
    - data_path (str): Directory path containing .npy files.
    
    Returns:
    - dict: Mapping of file names (without extension) to numpy arrays.
    """
    res_data = {}
    # Iterate over each file in the given directory.
    for doc in os.listdir(data_path):
        if doc.endswith('.npy'):
            # Load the numpy array and store it using the file name (without extension) as key.
            data = np.load(os.path.join(data_path, doc))
            res_data[doc.split('.npy')[0]] = data
    return res_data


# =============================================================================
# Function: flatten
# =============================================================================
def flatten(data):
    """
    Flatten matrices (or multi-dimensional arrays) to one dimension per sample.
    
    Parameters:
    - data (np.ndarray): Array with shape (num_samples, ...).
    
    Returns:
    - np.ndarray: Reshaped array with shape (num_samples, -1).
    """
    data = deepcopy(data)  # Deep copy to prevent modifying original data
    return data.reshape(len(data), -1)

# =============================================================================
# Class: BaseEncoder
# =============================================================================
class BaseEncoder:
    def __init__(self):
        """
        Base encoder initializer. This class serves as a template.
        """
        pass

    def transform(self, x, to_numpy=False, inverse=False):
        """
        Transform input data into encoded labels.
        
        Parameters:
        - x: Input data (can be a Pandas Series, list, or numpy array).
        - to_numpy (bool): If True, returns a numpy array.
        - inverse (bool): If True, performs the inverse transformation.
        
        Returns:
        - Transformed data in list or numpy array form.
        """
        if isinstance(x, pd.Series):
            return x.apply(lambda y: self.transform(y, to_numpy=to_numpy, inverse=inverse))
        
        if isinstance(x, np.ndarray) or isinstance(x, list):
            res_list = []
            for i in x:
                # Convert list or numpy array element to tuple if not doing inverse transformation.
                if not inverse and (isinstance(i, list) or isinstance(i, np.ndarray)):
                    i = tuple(i)
                res_list.append(self.encoder[i] if not inverse else self.decoder[i])
            return np.array(res_list) if to_numpy else res_list
        else:
            raise ValueError('No list-like inputs.')

    def single_transform(self, x, inverse=False):
        """
        Transform a single input element.
        
        Parameters:
        - x: A single data element (list/array will be converted to tuple).
        - inverse (bool): If True, performs the inverse transformation.
        
        Returns:
        - Encoded value for the input element.
        """
        if not inverse and (isinstance(x, list) or isinstance(x, np.ndarray)):
            x = tuple(x)
        return self.encoder[x] if not inverse else self.decoder[x]

# =============================================================================
# Class: LabelEncoder
# =============================================================================
class LabelEncoder(BaseEncoder):
    def __init__(self, vocab):
        """
        Initialize LabelEncoder with a vocabulary.
        
        Parameters:
        - vocab (iterable): Iterable of vocabulary items.
        """
        # Map each vocabulary item (converted to a tuple) to a unique integer.
        self.encoder = {
            tuple(word): idx + 1 for idx, word in enumerate(vocab)
        }
        self.vocab_size = max(self.encoder.values())
        # Create an inverse mapping from integers to vocabulary items.
        self.decoder = {v: k for k, v in self.encoder.items()}
    
    def get_vocab_size(self):
        """Return the size of the vocabulary."""
        return self.vocab_size
    
    def transform(self, x, to_numpy=False, inverse=False):
        return super().transform(x, to_numpy, inverse)
    
    def single_transform(self, x, inverse=False):
        return super().single_transform(x, inverse)

# =============================================================================
# Class: ImageLabelEncoder
# =============================================================================
class ImageLabelEncoder(BaseEncoder):
    def __init__(self, width, height, start_idx=1):
        """
        Initialize ImageLabelEncoder with image dimensions.
        
        Parameters:
        - width (int): Width of the image/matrix.
        - height (int): Height of the image/matrix.
        - start_idx (int): Starting index for encoding labels.
        """
        self.width = width
        self.height = height
        # Create a mapping from each coordinate (i, j) to a unique integer label.
        self.encoder = {
            (i, j): i * self.height + j + start_idx 
            for i in range(self.width) for j in range(self.height)
        }
        self.vocab_size = max(self.encoder.values())
        self.decoder = {v: k for k, v in self.encoder.items()}
    
    def get_vocab_size(self):
        """Return the vocabulary size for image encoding."""
        return self.vocab_size
    
    def matrix_transform(self, x, inverse=False):
        """
        Transform a matrix into a sequence of encoded labels based on nonzero coordinates.
        
        Parameters:
        - x (ndarray): Input matrix.
        - inverse (bool): If True, perform inverse transformation.
        
        Returns:
        - Encoded sequence.
        """
        coords = np.array(x.nonzero()).T  # Get coordinates of non-zero elements.
        return self.transform(coords, inverse=inverse)
    
    def transform(self, x, to_numpy=False, inverse=False):
        return super().transform(x, to_numpy, inverse)
    
    def single_transform(self, x, inverse=False):
        return super().single_transform(x, inverse)

# =============================================================================
# Function: create_matrix
# =============================================================================
def create_matrix(coords, resolution, global_x_min=None, global_x_max=None,
                  global_y_min=None, global_y_max=None, start_idx=1):
    """
    Create a discretized matrix from continuous coordinates.
    
    Parameters:
    - coords (ndarray): Array of shape (n, 2) with float coordinates.
    - resolution (float): Bin size for discretization.
    - global_x_min, global_x_max, global_y_min, global_y_max (float, optional):
      Global boundaries for x and y (if provided).
    - start_idx (int, optional): Starting index for labeling cells.
    
    Returns:
    - np.ndarray: Matrix with discretized coordinate values.
    """
    # Determine the boundary values using global settings or the min/max of coords.
    x_min = global_x_min if global_x_min is not None else coords[:, 0].min()
    x_max = global_x_max if global_x_max is not None else coords[:, 0].max()
    y_min = global_y_min if global_y_min is not None else coords[:, 1].min()
    y_max = global_y_max if global_y_max is not None else coords[:, 1].max()
    
    # Calculate the number of bins in each dimension.
    x_bins = int(np.ceil((x_max - x_min) / resolution)) + 1
    y_bins = int(np.ceil((y_max - y_min) / resolution)) + 1
    
    # Initialize a matrix filled with zeros.
    matrix = np.zeros((y_bins, x_bins))
    coord_set = set()  # To track which cells have been filled
    step = 0  # To assign incremental labels
    
    # Process each coordinate to determine its bin indices and fill the matrix.
    for x, y in coords:
        x_idx = int(np.floor((x - x_min) / resolution))
        y_idx = int(np.floor((y - y_min) / resolution))
        # Skip if the cell is already labeled.
        if (y_idx, x_idx) in coord_set:
            continue
        matrix[y_idx, x_idx] = step + start_idx  # Label the cell.
        coord_set.add((y_idx, x_idx))
        step += 1
    
    return matrix

def get_files(BASE_DIR='../itinerary_dataset'):
    """
    Retrieve file paths from subdirectories within BASE_DIR, excluding directories ending with '.git'.
    
    Parameters:
    - BASE_DIR (str): Base directory to search.
    
    Returns:
    - list: List of file paths.
    """
    files = []
    # Iterate through each item in the base directory.
    for dir in os.listdir(BASE_DIR):
        dir_path = os.path.join(BASE_DIR, dir)
        # Check if the item is a directory and is not a .git directory.
        if os.path.isdir(dir_path) and not dir.endswith('.git'):
            # Iterate through the files within the subdirectory.
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                files.append(file_path)
    return files
