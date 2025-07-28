import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from read_cifar import load_cifar


def convert_labels_to_is_vehicle(labels: np.ndarray) -> np.ndarray:
    vehicle_classes = np.array([0, 1, 8, 9])
    new_labels = np.isin(labels, vehicle_classes).astype(np.int64)
    return new_labels


def load_cifar_is_vehicle() -> tuple[dict, dict]:
    train_data, test_data = load_cifar()
    train_data["labels"] = convert_labels_to_is_vehicle(train_data["labels"])
    test_data["labels"] = convert_labels_to_is_vehicle(test_data["labels"])
    return train_data, test_data


def get_ith_entry(i: int, dataset: dict):
    img = dataset["data"][i]
    label = dataset["labels"][i]
    return img, label


def show_ith_entry(i: int, dataset: dict):
    img, label = get_ith_entry(i, dataset)
    plt.imshow(img)
    print(f"IsVehicle = {label}")


def preprocess_cifar_for_svc(cifar_dataset: dict[str, np.ndarray]):
    """Flattens CIFAR images into 1D arrays."""
    new_cifar_dataset = cifar_dataset.copy()

    image_count = new_cifar_dataset["data"].shape[0]
    new_cifar_dataset["data"] = new_cifar_dataset["data"].reshape(image_count, -1)
    return new_cifar_dataset


def split_cifar_data_and_labels(cifar_dataset: dict[str, np.ndarray]):
    data = cifar_dataset["data"]
    labels = cifar_dataset["labels"]
    return data, labels


def stratified_subset(
    data: np.ndarray, labels: np.ndarray, subset_size: int, random_seed: int = None
):
    """
    Get a stratified random subset of a classification dataset.

    Parameters:
    - data (np.ndarray): 2D array of shape (n_samples, n_features)
    - labels (np.ndarray): 1D array of integer labels with shape (n_samples,)
    - subset_size (int): Total number of samples in the subset
    - random_seed (int, optional): Seed for reproducibility

    Returns:
    - subset_data (np.ndarray): Subset of data with stratified class distribution
    - subset_labels (np.ndarray): Corresponding labels for the subset
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    unique_classes, counts = np.unique(labels, return_counts=True)

    # Calculate number of samples per class for the subset
    proportions = counts / counts.sum()
    samples_per_class = np.floor(proportions * subset_size).astype(int)

    # Adjust for rounding issues to match exact subset_size
    total_assigned = samples_per_class.sum()
    remainder = subset_size - total_assigned

    # Distribute remainder samples randomly among classes
    if remainder > 0:
        extra_indices = np.random.choice(len(unique_classes), remainder, replace=True)
        for idx in extra_indices:
            samples_per_class[idx] += 1

    subset_indices = []

    for cls, n_samples in zip(unique_classes, samples_per_class):
        class_indices = np.where(labels == cls)[0]
        selected_indices = np.random.choice(class_indices, n_samples, replace=False)
        subset_indices.extend(selected_indices)

    np.random.shuffle(subset_indices)

    subset_data = data[subset_indices]
    subset_labels = labels[subset_indices]

    return subset_data, subset_labels


def get_proportions_per_class(labels: np.ndarray):
    unique_classes, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    return {
        unique_class: proportion
        for unique_class, proportion in zip(unique_classes, proportions)
    }


def evaluate_classifier_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def cross_validation(X, y, parameter_search_space: dict, n_folds: int, model: SVC):
    cv_best_model = GridSearchCV(model, parameter_search_space, cv=n_folds)
    cv_best_model.fit(X, y)
    return cv_best_model
