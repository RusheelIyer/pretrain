import numpy as np

from src.datasets.foursquare import Foursquare
from src.datasets.movielens import Movielens_1m, Movielens_25m, Movielens_100k
from src.datasets.sap import Sap


def load_user_fea_dic(config, fea_type):
    """ TO BE DONE
    Args:
        config (dict): Dictionary of configuration
        fea_type (str): A string describing the feature type. Options:

    Returns:

    """
    pass


def load_item_fea_dic(config, fea_type):
    """ Load item feature

    Args:
        config (dict): Dictionary of configuration
        fea_type (str): A string describing the feature type. Options:
            - one_hot
            - word2vec
            - bert
            - cate
    Returns:
        dict: A dictionary with key being the item_id and value being the numpy array of feature vector

    """
    data_str = config["dataset"]
    root_dir = config["root_dir"]
    print("load basic item featrue for dataset:", data_str, " type:", fea_type)

    if fea_type == "word2vec":
        item_feature_file = open(
            root_dir + "datasets/" + data_str + "/raw/item_feature_w2v.csv", "r"
        )
    elif fea_type == "cate":
        item_feature_file = open(
            root_dir + "datasets/" + data_str + "/raw/item_feature_cate.csv", "r"
        )
    elif fea_type == "one_hot":
        item_feature_file = open(
            root_dir + "datasets/" + data_str + "/raw/item_feature_one.csv", "r"
        )
    elif fea_type == "bert":
        item_feature_file = open(
            root_dir + "datasets/" + data_str + "/raw/item_feature_bert.csv", "r"
        )
    else:
        print(
            "[ERROR]: CANNOT support other feature type, use 'random' user featrue instead!"
        )

    item_feature = {}
    lines = item_feature_file.readlines()
    for index in range(1, len(lines)):
        key_value = lines[index].split(",")
        item_id = int(key_value[0])
        feature = np.array(key_value[1].split(" "), dtype=np.float)
        item_feature[item_id] = feature
    return item_feature


def load_split_dataset(config):
    """Loading dataset

    Args:
        config (dict): Dictionary of configuration

    Returns:


    """
    dataset_mapping = {
        "ml_100k": Movielens_100k,
        "ml_1m": Movielens_1m,
        "ml_25m": Movielens_25m,
        "foursquare": Foursquare,
        "sap": Sap,
    }
    dataset = dataset_mapping[config["dataset"]]()
    return dataset.load_split(config)


def load_user_item_feature(config):
    """Loading features of users and items

    Args:
        config (dict): Dictionary of configuration

    Returns:
        user_feat (numpy.ndarray): The first column is the user id, rest column are feat vectors
        item_feat (numpy.ndarray): The first column is the itm id, rest column are feat vectors

    """
    dataset_mapping = {
        "ml_100k": Movielens_100k,
        "ml_1m": Movielens_1m,
        "ml_25m": Movielens_25m,
        "foursquare": Foursquare,
        "sap": Sap,
    }
    dataset = dataset_mapping[config["dataset"]]()
    return dataset.load_fea_vec()
