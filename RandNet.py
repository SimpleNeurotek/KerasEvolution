#!/usr/bin/env python
from keras.layers import Dense, Conv1D, Conv2D, LSTM, GRU, Flatten, Dropout, Input, Reshape, BatchNormalization
from keras.layers import MaxPool1D, MaxPool2D, Embedding, Bidirectional, TimeDistributed, concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Nadam, RMSprop, SGD
import keras.backend as K
import numpy as np
import os
import re
import pandas as pd
from random import choice
import seaborn as sns
from sklearn.utils.extmath import softmax
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook


### Hyperparameters for layer initializing - Begin###
optimizer_pool = {
        Adadelta: {'learning_rate': [1.0, 0.99, 0.95, 0.9, 0.85, 0.8]}, 
        Adagrad: {'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001]},
        Adam: {'learning_rate': [0.01, 0.005, 0.001, 0.0005, 0.0001]},
        Adamax: {'learning_rate': [0.02, 0.005, 0.002, 0.0005, 0.0002]},
        Nadam: {'learning_rate': [0.02, 0.005, 0.002, 0.0005, 0.0002]},
        RMSprop: {'learning_rate': [0.01, 0.005, 0.001, 0.0005, 0.0001]},
        SGD: {'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001], 
              'nesterov': [True], 'momentum': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}
}

LAYERS = {"Dense": Dense, "Conv1D": Conv1D, "Conv2D": Conv2D, 
          "LSTM": LSTM, "GRU": GRU, "Flatten": Flatten, 
          "Dropout": Dropout, "Input": Input, "Reshape": Reshape, 
          "BatchNormalization": BatchNormalization, "MaxPooling1D": MaxPool1D, 
          "MaxPooling2D": MaxPool2D, "Embedding": Embedding, 
          "Bidirectional": Bidirectional, "TimeDistributed": TimeDistributed}

parameter_pool = {
    "initializers": [
            "Zeros",
            "Ones",
            "RandomNormal",
            "RandomUniform",
            "TruncatedNormal",
            "VarianceScaling",
            "Orthogonal",
            "lecun_uniform",
            "glorot_normal",
            "glorot_uniform",
            "he_normal",
            "lecun_normal",
            "he_uniform"
        ],
    "activations": [
            "elu",
            "relu",
            "softmax",
            "selu",
            "softplus",
            "softsign",
            "tanh",
            "sigmoid",
            "hard_sigmoid",
            "exponential",
            "linear"
        ],
    "regularizers": [
            "l1",
            "l2",
            "l1_l2"
        ],
    
    "constraints": [
            "MaxNorm",
            "NonNeg",
            "UnitNorm",
            "MinMaxNorm"
        ]
    
}

layer_parameters = {
    "Dense": {
                "units": [2, 4, 6, 8, 10, 12,14, 16],
                "activation": parameter_pool["activations"], 
                "use_bias": [False, True], 
                "kernel_initializer": parameter_pool["initializers"], 
                "bias_initializer": parameter_pool["initializers"], 
                "kernel_regularizer": parameter_pool["regularizers"], 
                "bias_regularizer": parameter_pool["regularizers"], 
                "activity_regularizer": parameter_pool["regularizers"], 
                "kernel_constraint": parameter_pool["constraints"], 
                "bias_constraint": parameter_pool["constraints"]
            },
    "Conv1D": {
                "filters": [2, 4, 6, 8, 10, 12, 14, 16], 
                "kernel_size": [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,)], 
                "padding": ["same"], 
                # "data_format": [None, "channels_last", "channels_first"], 
                "activation": parameter_pool["activations"], 
                "use_bias": [True, False], 
                "kernel_initializer": parameter_pool["initializers"], 
                "bias_initializer": parameter_pool["initializers"], 
                "kernel_regularizer": parameter_pool["regularizers"], 
                "bias_regularizer": parameter_pool["regularizers"], 
                "activity_regularizer": parameter_pool["regularizers"], 
                "kernel_constraint": parameter_pool["constraints"], 
                "bias_constraint": parameter_pool["constraints"]
            },
    "Conv2D": {
                "filters": [2, 4, 6, 8, 10, 12, 14, 16], 
                "kernel_size": [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)], 
                # "padding": ["valid", "causal", "same"], 
                # "data_format": [None, "channels_last", "channels_first"], 
                "activation": parameter_pool["activations"], 
                "use_bias": [True, False], 
                "kernel_initializer": parameter_pool["initializers"], 
                "bias_initializer": parameter_pool["initializers"], 
                "kernel_regularizer": parameter_pool["regularizers"], 
                "bias_regularizer": parameter_pool["regularizers"], 
                "activity_regularizer": parameter_pool["regularizers"], 
                "kernel_constraint": parameter_pool["constraints"], 
                "bias_constraint": parameter_pool["constraints"]
            },
    "GRU": {
                "units": [2, 4, 6, 8, 10, 12, 14, 16], 
                "activation": parameter_pool["activations"], 
                "recurrent_activation": parameter_pool["activations"], 
                "use_bias": [True, False], 
                "kernel_initializer": parameter_pool["initializers"], 
                "recurrent_initializer": parameter_pool["initializers"], 
                "bias_initializer": parameter_pool["initializers"], 
                "kernel_regularizer": parameter_pool["regularizers"], 
                "recurrent_regularizer": parameter_pool["regularizers"], 
                "bias_regularizer": parameter_pool["regularizers"], 
                "activity_regularizer": parameter_pool["regularizers"], 
                "kernel_constraint": parameter_pool["constraints"], 
                "recurrent_constraint": parameter_pool["constraints"], 
                "bias_constraint": parameter_pool["constraints"], 
                "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 
                "recurrent_dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 
                "implementation": [1, 2], 
                "return_sequences": [True], 
                # "return_state": [True, False], 
                "go_backwards": [True, False], 
                # "stateful": [True, False], 
                "unroll": [True, False], 
                "reset_after": [True, False]
            },
    "LSTM": {
                "units": [2, 4, 6, 8, 10, 12, 14, 16], 
                "activation": parameter_pool["activations"], 
                "recurrent_activation": parameter_pool["activations"], 
                "use_bias": [True, False], 
                "kernel_initializer": parameter_pool["initializers"], 
                "recurrent_initializer": parameter_pool["initializers"], 
                "bias_initializer": parameter_pool["initializers"], 
                "unit_forget_bias": [True, False],
                "kernel_regularizer": parameter_pool["regularizers"], 
                "recurrent_regularizer": parameter_pool["regularizers"], 
                "bias_regularizer": parameter_pool["regularizers"], 
                "activity_regularizer": parameter_pool["regularizers"], 
                "kernel_constraint": parameter_pool["constraints"], 
                "recurrent_constraint": parameter_pool["constraints"], 
                "bias_constraint": parameter_pool["constraints"], 
                "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 
                "recurrent_dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 
                "implementation": [1, 2], 
                "return_sequences": [True], 
                # "return_state": [True, False], 
                "go_backwards": [True, False], 
                # "stateful": [True, False], 
                "unroll": [True, False], 
            },
    "Dropout": {
                "rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            },
    "Flatten": {
                "data_format": [None, "channels_last", "channels_first"]
            },
    "MaxPooling1D": {
                "pool_size": [2, 3, 4, 5, 6, 7, 8], 
                # "padding": ["valid", "causal", "same"],
                # "data_format": [None, "channels_last", "channels_first"]
            }, 
    "MaxPooling2D": {
                "pool_size": [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)], 
                # "padding": ["valid", "causal", "same"],
                # "data_format": [None, "channels_last", "channels_first"]
            } 
}

def get_random_layer_parameters(layer):
            layer_name = re.sub(r'<.+\'(.+)\'>', r'\1', str(layer)).split(".")[-1]
            random_parameters = {key: choice(value) for key, value in layer_parameters[layer_name].items()}
            return layer(**random_parameters)
            
### Hyperparameters for layer initializing - End###


class RandNet(object):
    
    def __init__(self, feature_data_shape, label_data_shape,
                 num_classes=1,
                 get_random_layer_parameters=get_random_layer_parameters,
                 optimizer_pool=optimizer_pool,
                 layer_types=LAYERS, init_hl=True):
        
        """
        Creates a randomly initialized neural net in keras with 
        functional API Model and is used for initial state for
        a modified NEAT-algorithm
        
        parameters:
        - feature_data_shape: tuple, shape of the feature data
        - label_data_shape: tuple, shape of the label data 
                            -> NO one-hot-encoding needed
        => easiest way: feature_data / label_data as numpy array 
           -> use shape attribute
        - num_classes: int, number of classes in label data
           -> defaults to: 1 -> binary classification
        - get_random_layer_parameters: function, randomly initializes layer
        - optimizer_pool: dict, parameters for keras optimizers
        """
        
        # data configs
        self.feature_data_shape = feature_data_shape
        self.label_data_shape = label_data_shape
        self.feature_shape_length = len(self.feature_data_shape)
        self.input_shape = tuple(list(self.feature_data_shape)[1:])
        self.output_shape = tuple(list(self.label_data_shape)[1:])
        self.num_classes = num_classes
        self.layer_names = [str(lt) for lt in layer_types.values()]
        self.layer_types = list(layer_types.values())
        self.core_layers = [Dense, Conv1D, Conv2D, LSTM, GRU]
        self.optimizer_pool = optimizer_pool
        self.get_random_layer_parameters = get_random_layer_parameters
        self.optimizers = ["sgd", "rmsprop", "adadelta", "adam", 
                           "adamax", "nadam"]
               
        # init net
        self.input_layer = Input(shape=self.input_shape)
        if init_hl:
            self.hidden_layers = pd.DataFrame(data={"parallel_1": \
                                                    [get_random_layer_parameters(choice(self.core_layers))]})
        else:
            self.hidden_layers = pd.DataFrame(data={"parallel_1": \
                                                    []})
        if num_classes is None or 1 <= num_classes <= 2:
            self.output_layer = Dense(1)
        else:
            self.output_layer = Dense(num_classes)
        self.output_tensors = []
        self.insertion_count = 0
        self.deletion_count = 0
        self.mutation_count = 0
        self.input_tensor = self.input_layer
        self.output_tensor = None
        
    def initialize_random_layer(self, layers=[Dense, Conv1D, Conv2D, LSTM, GRU]):
        return self.get_random_layer_parameters(choice(layers))
        
    def insertion(self, layer, axis=0):
        
        cols = self.hidden_layers.columns
        df_length = len(self.hidden_layers)
        if axis == 0:
            # vertical insertion
            rand_col = choice(cols)
            if isinstance(layer, tuple(self.core_layers)):
                self.hidden_layers.loc[df_length, rand_col] = layer
            else:
                self.hidden_layers.loc[df_length, rand_col] = self.get_random_layer_parameters(layer)
            
        elif axis == 1:
            # horizontal insertion
            parallel_count = re.search(r'_(\d+)', cols[-1])
            if parallel_count:
                parallel_count = int(parallel_count.group(1))
            else:
                raise TypeError("No column number found in column string.")
            
            if isinstance(layer, tuple(self.core_layers)):
                self.hidden_layers["parallel_{}".format(parallel_count + 1)] = \
                [layer] + ([np.nan] * (df_length - 1))
            else:
                self.hidden_layers["parallel_{}".format(parallel_count + 1)] = \
                [self.get_random_layer_parameters(layer)] + ([np.nan] * (df_length - 1))            
            
        else:
            raise ValueError("Axis-Error: please check the value of the axis parameter.")
            
    def modify(self, layer, input_tensor):
        # insert some modification layers like MaxPooling etc.
        rand_num = np.random.uniform(0, 1)
        layer_name = str(type(layer))
        
        # some maxpooling
        if layer_name == str(Conv1D):
            output_tensor = self.get_random_layer_parameters(MaxPool1D)(input_tensor)
        elif layer_name == str(Conv2D) and len(input_tensor.shape) == 3:
            output_tensor = self.get_random_layer_parameters(MaxPool1D)(input_tensor)
        elif layer_name == str(Conv2D) and len(input_tensor.shape) > 3:
            output_tensor = self.get_random_layer_parameters(MaxPool2D)(input_tensor)
        else:
            output_tensor = input_tensor
            
        # some Dropout layer
        if rand_num <= 0.5:
            output_tensor = self.get_random_layer_parameters(Dropout)(output_tensor)
        # some BatchNormalization layer
        elif 0.5 < rand_num <= 0.75:
            output_tensor = BatchNormalization()(output_tensor)
            
        return output_tensor
    
    def bidirectional(self, recurrent_layer):
        # make a bidirectional connection to a recurrent layer
        return Bidirectional(recurrent_layer)
    
    def timedistribute(self, dense_layer):
        # make a timedistributed connection to a dense layer
        return TimeDistributed(dense_layer)
    
    def send_tensor_and_reshape(self, modify=True):
        # walk through the hidden_layers dataframe and 
        # connect the layers with the input tensor and
        # insert reshape layer if necessary
        output_tensors = []
        for parallel in self.hidden_layers.columns:
            tensor = None
            for i, layer in enumerate(self.hidden_layers[parallel].values.tolist()):
                if str(layer) == 'nan':
                    continue
                if i == 0:
                    tensor = self.inject_reshape(layer, self.input_layer)
                else:
                    tensor = self.inject_reshape(layer, tensor)
                # add some modification layers
                if modify:
                    if np.random.uniform(0, 1) <= 0.5:
                        tensor = self.modify(layer, tensor)
            if tensor is not None:
                output_tensors.append(tensor)
        self.output_tensors = output_tensors
        
    def concat_output_tensors(self):
        # concatenate the output tensors to one 
        # final tensor for last dense layer
        reshaped_tensors = []
        for ot in self.output_tensors:
            try:
                reshaped_tensors.append(Reshape((-1,))(ot))
            except:
                continue
        if len(reshaped_tensors) >= 2:
            return concatenate(reshaped_tensors)
        elif len(reshaped_tensors) == 1:
            return reshaped_tensors[0]
        else:
            return reshaped_tensors
                    
    def inject_reshape(self, layer, input_tensor):
        layer_type = str(type(layer))
        tensor_shape = input_tensor.shape
        tensor_shape_length = len(tensor_shape)
        if layer_type == str(Dense):
            return layer(input_tensor)
        elif layer_type in [str(LSTM), str(GRU), str(Conv1D)]:
            # (batch_size, features, time_steps)
            if tensor_shape_length == 3:
                return layer(input_tensor)
            elif tensor_shape_length == 2:
                reshaped = Reshape((tensor_shape[1], 1))(input_tensor)
                return layer(reshaped)
            elif tensor_shape_length > 3:
                reshaped = Reshape((tensor_shape[1], -1))
                return layer(reshaped)
            else:
                raise ValueError("The Input tensor must have more than 2 dimensions for recurrent layers."\
                                 "Given:  {}.".format(tensor_shape))
        elif layer_type == str(Conv2D):
            if tensor_shape_length == 2:
                new_layer = self.get_random_layer_parameters(Conv1D)
                reshaped = Reshape((tensor_shape[1], 1))(input_tensor)
                print("2 dim to Conv1D: " + str(reshaped.shape))
                return new_layer(reshaped)
            elif tensor_shape_length == 3:
                reshaped = Reshape((tensor_shape[1], tensor_shape[2], 1))(input_tensor)
                print("3 dim to 4 dim: " + str(reshaped.shape))
                return layer(reshaped)
            elif tensor_shape_length >= 4:
                reshaped = Reshape((tensor_shape[1], tensor_shape[2], -1))(input_tensor)
                return layer(reshaped)
            else:
                raise ValueError("The Input tensor must have more than 3 dimensions for Conv2D layers."\
                                 "Given:  {}.".format(tensor_shape))
                    
    
    def create_model(self, print_summary=True):
        # randomly pick optimizer
        opti = choice(list(self.optimizer_pool.keys()))
        opti = opti(**{key: choice(value) for key, value in self.optimizer_pool[opti].items()})
        
        # connect net parts
        self.send_tensor_and_reshape()
        self.output_tensor = self.output_layer(self.concat_output_tensors())
        
        # initialize model
        model = Model(inputs=self.input_tensor, outputs=self.output_tensor)
        if print_summary:
            model.summary()
        if self.num_classes > 2:
            model.compile(optimizer=opti, 
                          loss="sparse_categorical_crossentropy", 
                          metrics=["sparse_categorical_accuracy"])
        elif self.num_classes in [1, 2]:
            model.compile(optimizer=opti, 
                          loss="binary_crossentropy", 
                          metrics=["binary_accuracy"])
        else:
            raise ValueError("Some Error in choosing the loss function ...")

        return model
        
        
if __name__ == '__main__':
    
    # loading breast cancer data
    bc_data = load_breast_cancer()
    bc_features = bc_data['data']
    bc_labels = bc_data['target']
    X_train, X_test, y_train, y_test = train_test_split(bc_features, bc_labels, test_size=0.2, random_state=42)
    
    # Testing RandNet class
    randnet = RandNet(feature_data_shape=X_train.shape, label_data_shape=y_train.shape)
    randnet.insertion(layer=choice(randnet.core_layers), axis=0)
    randnet.insertion(layer=choice(randnet.core_layers), axis=1)
    print(randnet.hidden_layers)
    model = randnet.create_model()
    model.fit(X_train, y_train, batch_size=int(round(X_train.shape[0] * 0.05)), epochs=10, validation_data=(X_test, y_test))
    del randnet
    del model