import os
import sys
parent_dir = os.path.join(os.getcwd(), '..')
if parent_dir not in sys.path: sys.path.append(parent_dir)
from utility.data_utils import *
from utility.visuals import *
from model.dimensional_reduction import *
from model.gans import *
from model.rf import *
from model.dnn import *
from model.knn import *
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
PCA_N_COMPONENTS = float(os.getenv("PCA_N_COMPONENTS"))
NOISE_DIM = int(os.getenv("NOISE_DIM"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS"))
NUM_EPOCHS = 7000


files = get_files(DATA_DIR)
for file in files:
    city_name = file.split('\\')[-1].split('.csv')[0].split('_')[0]

    path_name = process_and_transform_data(file, resolution=.5, overwrite=True)
    with open(path_name, 'rb') as f:
        result_dict = pickle.load(f)
    labels = result_dict['labels']
    encoder = result_dict['encoder']
    MAX_LEN = result_dict['max length']
    file_name = result_dict['file name']
    WIDTH = result_dict['width']
    HEIGHT = result_dict['height']
    unique_labels = [u for u in labels if np.array(np.where(u != 0)).T.shape[0] > 1]
    train_labels, test_labels = train_test_split(np.expand_dims(np.array(unique_labels), axis=1), test_size=.2)
    dr = DimensionalityReducer(
        method='tsne'
    )
    transformed_tsne = dr.fit_and_transform(train_labels.reshape(train_labels.shape[0], -1))
    
    knn = KNNFeaturesExtractor()
    knn.fit(transformed_tsne)
    features = knn.predict(transformed_tsne)
    features = features.reshape(len(features), -1)
    OUTPUT_DIM = transformed_tsne.shape[1]
    CONDITION_DIM = features.shape[-1]
    x, y, _, _ = get_x_y(test_labels, MAX_LEN=MAX_LEN, encoder=encoder)
    x = np.array(x)
    y = np.array(y)
    test_pca = PCATransformer(n_components=transformed_tsne.shape[-1])
    test_transformed = test_pca.fit_and_predict(
    x.reshape(x.shape[0], -1)
    )
    test_features = knn.predict(test_transformed)
    test_features = test_features.reshape(len(test_features), -1)

    knn_manager = KNNManager(n_neighbors=1)
    knn_manager.train(features, transformed_tsne)
        
    dnn_manager = DNNManager(
        input_dim=CONDITION_DIM, 
        output_dim=OUTPUT_DIM, 
        batch_size=BATCH_SIZE, 
        num_epochs=NUM_EPOCHS,
        lr=3e-4,
        betas=(.5, 0.999)
    )
    dnn_manager.train(features, transformed_tsne, verbose=1)

    gan_manager = GANManager(
        noise_dim=NOISE_DIM,
        condition_dim=features.shape[-1],
        output_dim=transformed_tsne.shape[1],
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        lr=0.0002,
        betas = (.5, 0.999),
    )
    gan_manager.train(features, transformed_tsne, verbose=1)

    rf_manager = RandomForestManager()
    rf_manager.train(features, transformed_tsne)

    knn_manager = KNNManager(n_neighbors=1)
    knn_manager.train(features, transformed_tsne)

    gan_output = gan_manager.predict(test_features)
    gan_output_shaped = test_pca.predict(gan_output, inverse=True)\
        .reshape(len(gan_output), x.shape[-2], x.shape[-1])
    gan_output_shaped[gan_output_shaped < 0] = 0

    dnn_output = dnn_manager.predict(test_features)
    dnn_output_shaped = test_pca.predict(dnn_output, inverse=True)\
        .reshape(len(dnn_output), x.shape[-2], x.shape[-1])
    dnn_output_shaped[dnn_output_shaped < 0] = 0

    rf_output = rf_manager.predict(test_features)
    rf_output_shaped = test_pca.predict(rf_output, inverse=True)\
        .reshape(len(rf_output), x.shape[-2], x.shape[-1])
    rf_output_shaped[rf_output_shaped < 0] = 0

    knn_output = knn_manager.predict(test_features)
    knn_output_shaped = test_pca.predict(knn_output, inverse=True)\
        .reshape(len(knn_output), x.shape[-2], x.shape[-1])
    knn_output_shaped[knn_output_shaped < 0] = 0
    
    SIGMA = 1
    # Apply Gaussian filter to each 2D image in a set of images
    def apply_gaussian(images, sigma):
        # images: numpy array with shape (n_samples, height, width)
        return np.array([gaussian_filter(img, sigma=sigma) for img in images])

    # Apply filter to each model's output
    gan_filtered = apply_gaussian(gan_output_shaped, sigma=SIGMA)
    dnn_filtered = apply_gaussian(dnn_output_shaped, sigma=SIGMA)
    rf_filtered  = apply_gaussian(rf_output_shaped, sigma=SIGMA)
    knn_filtered = apply_gaussian(knn_output_shaped, sigma=SIGMA)
    act = x + y
    act = act[:, 0, :]
    # Ensure the ground truth 'act' is filtered per image (if not already)
    act_filtered = np.array([gaussian_filter(img, sigma=SIGMA) for img in act])




    # Calculate average metrics for each model
    jsd_gan = np.mean([compute_jsd(act_filtered[i], gan_filtered[i]) for i in range(len(act_filtered))])
    jsd_dnn = np.mean([compute_jsd(act_filtered[i], dnn_filtered[i]) for i in range(len(act_filtered))])
    jsd_rf  = np.mean([compute_jsd(act_filtered[i], rf_filtered[i]) for i in range(len(act_filtered))])
    jsd_knn = np.mean([compute_jsd(act_filtered[i], knn_filtered[i]) for i in range(len(act_filtered))])

    rmse_gan = np.mean([compute_rmse(act_filtered[i], gan_filtered[i]) for i in range(len(act_filtered))])
    rmse_dnn = np.mean([compute_rmse(act_filtered[i], dnn_filtered[i]) for i in range(len(act_filtered))])
    rmse_rf  = np.mean([compute_rmse(act_filtered[i], rf_filtered[i]) for i in range(len(act_filtered))])
    rmse_knn = np.mean([compute_rmse(act_filtered[i], knn_filtered[i]) for i in range(len(act_filtered))])

    # Create a DataFrame to store the evaluation metrics
    results_df = pd.DataFrame({
        "Model": ["GAN", "DNN", "RF", "KNN"],
        "Jensen-Shannon Divergence": [jsd_gan, jsd_dnn, jsd_rf, jsd_knn],
        "RMSE": [rmse_gan, rmse_dnn, rmse_rf, rmse_knn],
        "Filtered Output": [gan_filtered, dnn_filtered, rf_filtered, knn_filtered]
    })

    print(results_df)
