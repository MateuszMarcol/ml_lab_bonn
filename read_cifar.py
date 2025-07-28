import pickle
import numpy as np


def load_cifar() -> tuple[dict, dict]:
    datapath = "data/cifar-10-batches-py"
    training_filepaths = [datapath + "/" + "data_batch_" + str(i) for i in range(1, 6)]
    test_filepaths = [datapath + "/" + "test_batch"]

    training_data = load_from_filepaths(training_filepaths)
    test_data = load_from_filepaths(test_filepaths)

    training_data["data"] = reshape_images(training_data["data"])
    test_data["data"] = reshape_images(test_data["data"])

    return training_data, test_data


def load_from_filepaths(filepaths: list[str]) -> dict:
    data_dict = dict()
    data_dict["data"] = []
    data_dict["labels"] = []

    data_arrays = []

    for name in filepaths:
        batch_dict = load_file(name)
        data_arrays.append(batch_dict["data".encode("UTF-8")])
        data_dict["labels"].extend(batch_dict["labels".encode("UTF-8")])

    data_dict["labels"] = np.array(data_dict["labels"])
    data_dict["data"] = np.concatenate(data_arrays)

    return data_dict


def load_file(file) -> dict:
    with open(file, "rb") as fo:
        data_dict = pickle.load(fo, encoding="bytes")
    return data_dict


def reshape_images(images_1d: np.ndarray):
    # images_1d.shape = (#imgs, 1024*3)
    red = images_1d[:, 0:1024].reshape(-1, 32, 32)
    green = images_1d[:, 1024:2048].reshape(-1, 32, 32)
    blue = images_1d[:, 2048:].reshape(-1, 32, 32)
    image_rgb = np.stack([red, green, blue], axis=-1)
    return image_rgb
