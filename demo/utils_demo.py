from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt

def get_data(num_outliers=150):
    # Setting for dataset
    tensor_shape = (90,90,2)
    num_points = 5000
    noise = 0.07
    flip_fraction = 0.5 
    
    # generate data points
    X, y = make_moons(n_samples=num_points, noise=noise)
    X_rot = np.column_stack([-X[:, 1], X[:, 0]])  # rotate 
    X_min = X_rot.min(axis=0)
    X_max = X_rot.max(axis=0)
    X_scaled = (X_rot - X_min) / (X_max - X_min)
    X_idx = (X_scaled * (np.array(tensor_shape[:2]) - 1)).astype(int)
    
    # initalize tensor
    tensor = np.zeros(tensor_shape)
    for (i, j), label in zip(X_idx, y):
        tensor[i, j, label] += 1
    
    # --- Two kinds of outliers ---
    if num_outliers > 0:
        num_flip = int(num_outliers * flip_fraction)
        num_spatial = num_outliers - num_flip
    
        # --- 1. label fliped ---
        idx_flip = np.random.choice(len(X_idx), size=num_flip, replace=False)
        for idx in idx_flip:
            i, j = X_idx[idx]
            wrong_label = 1 - y[idx]  # mis label
            tensor[i, j, wrong_label] += 1
    
        # --- 2. noise on corners ---
        corner_regions = [
            ([0.0, 0.2], [0.0, 0.2]),  # left down
            ([0.8, 1.0], [0.0, 0.2]),  # right up
            ([0.0, 0.2], [0.8, 1.0]),  # left up
            ([0.8, 1.0], [0.8, 1.0]),  # right down
        ]
    
        outlier_scaled = []
        for _ in range(num_spatial):
            x_range, y_range = corner_regions[np.random.randint(0, 4)]
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            outlier_scaled.append([x, y])
    
        outlier_scaled = np.array(outlier_scaled)
        outlier_idx = (outlier_scaled * (np.array(tensor_shape[:2]) - 1)).astype(int)
        outlier_labels = np.random.randint(0, 2, size=len(outlier_idx))
    
        for (i, j), label in zip(outlier_idx, outlier_labels):
            tensor[i, j, label] += 1
    return tensor

import matplotlib.patches as patches

def get_plot(tensor, scaler_tensor, gamma=0.1, title="", ax=None,
                                 fontsize=25, fontname='DejaVu Sans',
                                 highlight_rect=None):  # highlight_rect=(x, y, width, height)
    tensor_shape = np.shape(tensor)
    rgb_image = np.zeros((tensor_shape[0], tensor_shape[1], 3), dtype=np.uint8)
    
    if scaler_tensor[:, :, 0].max() > 0:
        red = (tensor[:, :, 0] / scaler_tensor[:, :, 0].max()) ** gamma
        rgb_image[:, :, 0] = (red * 255).astype(np.uint8)
    
    if scaler_tensor[:, :, 1].max() > 0:
        blue = (tensor[:, :, 1] / scaler_tensor[:, :, 1].max()) ** gamma
        rgb_image[:, :, 2] = (blue * 255).astype(np.uint8)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb_image)
    ax.set_title(title, fontsize=fontsize, fontname=fontname)
    ax.axis("off")
    
    if highlight_rect is not None:
        x, y, w, h = highlight_rect
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor='yellow', facecolor='none'
        )
        ax.add_patch(rect)
        
    if highlight_rect is not None:
        rect = patches.Rectangle(
            (0, 70), 19, 19,
            linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)

    if highlight_rect is not None:
        rect = patches.Rectangle(
            (70, 0), 19, 19,
            linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)


    return ax
