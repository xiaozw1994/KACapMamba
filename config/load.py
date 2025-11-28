import numpy as np
import os
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, Any


def ReadArrfFiles(filename):
    data = arff.loadarff(filename)
    df = pd.DataFrame(data[0])
    index = df.columns
    features = df[index[0]]
    label = df[index[1]]
    trains_feature = []
    trains_label = []
    for na in label:
        if na == b'n':
            trains_label.append(0)
        elif na == b's':
            trains_label.append(1)
        else:
            trains_label.append(2)
    trains_label = np.array(trains_label)
    for j in range(len(features)):
        tmp = []
        for i in range(len(features[j])):
            each_feature = np.array(list(features[j][i]))
            #print(each_feature)
            if np.isnan(each_feature).any():
                c = each_feature
                c[np.isnan(c)] = np.nanmean(c)
                tmp.append(c)
            else:
                tmp.append(each_feature)
        tmp = np.array(tmp)
        trains_feature.append(tmp)
    trains_feature = np.array(trains_feature)
    trains_feature = trains_feature.transpose((0,2,1))
    return trains_feature,trains_label


def read_uea_dataset(
    dataset_name: str, 
    root_dir: str = '../data/',
    fill_strategy: str = 'column_mean'  # Filling strategy: column_mean(column mean)/sample_mean(sample mean)
) -> Dict[str, Any]:
    """
    Read UEA dataset and automatically detect and fill NaN/Inf and other abnormal values
    
    Parameters:
        dataset_name: str, Dataset name
        root_dir: str, Dataset root directory
        fill_strategy: str, Filling strategy: 'column_mean' fill by column mean, 'sample_mean' fill by sample mean
    
    Returns:
        dict: Dictionary containing processed training/test set features and labels
    """
    def _fill_abnormal_values(data: np.ndarray, strategy: str = 'column_mean') -> np.ndarray:
        """Detect and fill abnormal values (NaN/Inf)"""
        # Convert to float32 type
        data = data.astype(np.float32)
        
        # Check for abnormal values
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        
        if not has_nan and not has_inf:
            return data  # No abnormal values, return directly
        
        print(f"Detected abnormal values (NaN/Inf), filling with {strategy} strategy...")
        
        # Convert Inf to NaN for unified processing
        data = np.where(np.isinf(data), np.nan, data)
        
        if strategy == 'column_mean':
            # Fill by column (feature dimension) mean
            if len(data.shape) == 3:
                # 3D data: (number of samples, time steps, number of variables)
                for var in range(data.shape[2]):
                    col_means = np.nanmean(data[:, :, var], axis=0)
                    for step in range(data.shape[1]):
                        nan_mask = np.isnan(data[:, step, var])
                        if np.any(nan_mask):
                            data[nan_mask, step, var] = col_means[step]
            else:
                # 2D data: (number of samples, number of features)
                col_means = np.nanmean(data, axis=0)
                for col in range(data.shape[1]):
                    nan_mask = np.isnan(data[:, col])
                    if np.any(nan_mask):
                        data[nan_mask, col] = col_means[col]
        
        elif strategy == 'sample_mean':
            # Fill by sample mean
            if len(data.shape) == 3:
                # 3D data: (number of samples, time steps, number of variables)
                for sample in range(data.shape[0]):
                    sample_mean = np.nanmean(data[sample])
                    nan_mask = np.isnan(data[sample])
                    if np.any(nan_mask):
                        data[sample, nan_mask] = sample_mean
            else:
                # 2D data: (number of samples, number of features)
                row_means = np.nanmean(data, axis=1)
                for row in range(data.shape[0]):
                    nan_mask = np.isnan(data[row])
                    if np.any(nan_mask):
                        data[row, nan_mask] = row_means[row]
        
        return data

    # Concatenate paths
    train_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_TRAIN.arff")
    test_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_TEST.arff")
    
    # Read data
    X_train, Y_train = ReadArrfFiles(train_path)
    X_test, Y_test = ReadArrfFiles(test_path)
    
    # Detect and fill abnormal values
    X_train = _fill_abnormal_values(X_train, fill_strategy)
    X_test = _fill_abnormal_values(X_test, fill_strategy)
    
    # Verify filling results
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        print("Warning: Abnormal values still exist in training set!")
    if np.isnan(X_test).any() or np.isinf(X_test).any():
        print("Warning: Abnormal values still exist in test set!")
    
    return {
        'TrainX': X_train,
        'TrainY': Y_train,
        'TestX': X_test,
        'TestY': Y_test
    }









