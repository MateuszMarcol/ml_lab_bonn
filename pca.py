import numpy as np
from sklearn.decomposition import PCA
from read_cifar import load_cifar



def vectorize_data(training_data, normalize=True):
    # vectorize the training data
    vectorized_training_data = training_data.reshape(training_data.shape[0], -1)

    if normalize:
        # normalize the training data
        vectorized_training_data = vectorized_training_data / 255
    return vectorized_training_data


# perform PCA
def compute_PCA(training_data: np.ndarray, number_of_components: int):
    training_data = vectorize_data(training_data)
    PCA_model = PCA(number_of_components)
    PCA_model.fit(training_data)
    return PCA_model