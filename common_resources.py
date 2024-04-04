import constant as co
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def neural_network_blue_print(
    input_node_count,
    activation_fnc,
    output_node_count,
    hidden_layer_count=1,
    hidden_layer_node_count=10,
    hidden_layer_activation=None,
    output_layer_activation=None,
):
    if [input_node_count, activation_fnc, output_node_count].__contains__(None):
        raise Exception(
            f"[input_node_count,activation_fnc,output_node_count] : {str([input_node_count,activation_fnc,output_node_count])}  should not be None"
        )
    blue_print = {
        co.INPUT_NODE_COUNT: input_node_count,
        co.ACTIVATION_FUNCTION: activation_fnc,
        co.OUTPUT_NODE_COUNT: output_node_count,
        co.HIDDEN_LAYER_COUNT: hidden_layer_count,
        co.HIDDEN_LAYER_NODE_COUNT: hidden_layer_node_count,
    }
    if hidden_layer_activation is not None:
        blue_print[co.ACTIVATION_FUNCTION_HIDDEN_LAYER] = hidden_layer_activation
    if output_layer_activation is not None:
        blue_print[co.ACTIVATION_FUNCTION_OUTPUT_LAYER] = output_layer_activation
    return blue_print


def load_data(path):
    return pd.read_csv(path)


def refactor_input_out_put_parameter(data):
    vectorizer = TfidfVectorizer()
    label_encoder = LabelEncoder()

    out_put_parameter = {
        "Resolved": 0,
        "Deleted": 1,
        "open": 2,
        "closed": 3,
        "Resolved & Closed": 4,
        "paused": 5,
    }

    # data["status"] = data["status"].apply(
    #     lambda x: torch.tensor(out_put_parameter.get(x), dtype=torch.long)
    # )
    # data["text"] = data["text"].apply(
    #     lambda x: torch.tensor(
    #         vectorizer.fit_transform([x]).toarray(), dtype=torch.float32
    #     )
    # )
    out_label = data["status"].tolist()
    input_lable = data["text"].tolist()
    x_train = torch.tensor(
        vectorizer.fit_transform(input_lable).toarray(), dtype=torch.float32
    )
    y_train = torch.tensor(label_encoder.fit_transform(out_label), dtype=torch.long)
    return x_train, y_train


def suffeling_date(data):
    data = data.sample(frac=1).reset_index(drop=True)
    return data


def split_train_test_dataset(x_train, y_train, trainig_rate):
    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=trainig_rate, random_state=42
    )

    return x_train, x_test, y_train, y_test


def train_data_set_split_batch(x_train, y_train, batch_size):

    train_data = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader
