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


class EvoNet(object):
    
    def __init__(self, feature_data, label_data, start_population=10, 
                 num_classes=1, threshold=0.99, max_generations=10, layer_dict=LAYERS):
        
        self.breeding_pool = {}
        self.feature_data = np.array(feature_data)
        self.label_data = np.array(label_data)
        self.start_population = start_population
        self.num_classes = num_classes
        self.data_amount = self.feature_data.shape[0]
        self.gen_count = 1
        self.offspring_count = self.start_population - 2
        self.max_generations = max_generations
        self.threshold = threshold
        self.layer_dict = layer_dict
        self.core_layers = [Dense, Conv1D, Conv2D, LSTM, GRU]
        # batch_size depending on amount of data
        self.batch_size = round(0.05 * self.data_amount)
            
        # train and test data split
        if self.data_amount < 1000:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.feature_data, self.label_data, test_size=0.33, random_state=42)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.feature_data, self.label_data, test_size=0.2, random_state=42)
        
    def generate_initial_generation(self):
        key = "gen_1"
        first_population = []
        for i in range(self.start_population):
            # generate randnets for modelling
            randnet = RandNet(self.feature_data.shape, self.label_data.shape, num_classes=self.num_classes)
            first_population.append({"model": randnet.create_model(), "checkpoint": None, "train_history": None})
                    
        self.breeding_pool[key] = first_population
    
    def mutate(self):
        pass
    
    def evaluate(self):
        """
        Train each model over 10 epochs and choose best 
        accuracy for ranking.
        Appends top 2 tuple at gen value (population) in dict
        """
        directory = ".model_results/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        for gen, population in self.breeding_pool.items():
            if population[0]["checkpoint"] is None:
                for individuum in population:
                    checkpoint = ModelCheckpoint(directory + "{}_{}.h5".format(gen, individuum["model"].name), 
                                                 verbose=False, monitor=individuum["model"]._compile_metrics[0],
                                                 save_best_only=True, mode='max')
                    individuum["train_history"] = individuum["model"].fit(self.X_train, self.y_train, epochs=10, 
                                                                          validation_data=(self.X_test, self.y_test), 
                                                                          callbacks=[checkpoint], verbose=False)
                    individuum["checkpoint"] = checkpoint
                best_epochs = sorted([individuum["checkpoint"].best for individuum in population], reverse=True)
                top_2 = tuple(sorted([individuum for individuum in population if individuum["checkpoint"].best in best_epochs[:2]], key=lambda x: x["checkpoint"].best, reverse=True))
                self.breeding_pool[gen].append(top_2)
    
    def mate(self):
        previous_key = "gen_{}".format(self.gen_count)
        previous_gen = self.breeding_pool[previous_key]
        new_key = "gen_{}".format(self.gen_count + 1)
        top_2 = self.breeding_pool[previous_key][-1]
        male, female = (top_2[0]["model"], top_2[1]["model"])
        male_hidden_layers = self.get_hidden_layers(male)
        female_hidden_layers = self.get_hidden_layers(female)
        
        if self.gen_count == 1:
            
            rand_num_1 = np.random.uniform(0, 1)
            # offspring
            offsprings = []
            for j in tqdm_notebook(range(self.offspring_count)):
                for z in range(5):
                    try:
                        randnet = RandNet(self.feature_data.shape, self.label_data.shape, 
                                          num_classes=self.num_classes, init_hl=True)

                        if rand_num_1 <= 0.5:
                            rand_num_2 = np.random.uniform(0, 1)
                            # insert male hidden layers
                            self.insert_hidden_layers(randnet, male_hidden_layers)
                            # insert female hidden layers
                            if rand_num_2 <= 0.5:
                                self.insert_hidden_layers(randnet, female_hidden_layers)
                            else:
                                self.insert_hidden_layers(randnet, female_hidden_layers, axis=1)
                            offsprings.append({"model": randnet.create_model(), "checkpoint": None, "train_history": None})
                        else:
                            rand_num_2 = np.random.uniform(0, 1)
                            # insert male hidden layers
                            self.insert_hidden_layers(randnet, female_hidden_layers)
                            # insert female hidden layers
                            if rand_num_2 <= 0.5:
                                self.insert_hidden_layers(randnet, male_hidden_layers)
                            else:
                                self.insert_hidden_layers(randnet, male_hidden_layers, axis=1)
                            offsprings.append({"model": randnet.create_model(), "checkpoint": None, "train_history": None})
                        break
                    except ValueError:
                        continue
                # adding offspring to new generation in breeding_pool 
                self.breeding_pool[new_key] = offsprings
        else:
            # define some layer mating and crossing over
            pass
        
        self.gen_count += 1
        
    def insert_hidden_layers(self, randnet, hidden_layers, axis=0, modify=False):
        for hl in hidden_layers:
            randnet.insertion(hl, axis)
            
    # randnet as parameter instead of model    
    def get_hidden_layers(self, model):
        model_layers = model.get_config()["layers"]
        model_hidden_layers = model_layers[1:-1]
        recovered_layers = []
        for hidden_layer in model_hidden_layers:
            new_layer = self.layer_dict[hidden_layer["class_name"]].from_config(hidden_layer["config"])
            if str(type(new_layer)) in [str(cl) for cl in self.core_layers]:
                recovered_layers.append(new_layer)
            
        return recovered_layers
        
    def layer_class_to_class_name(layer_class):
        return re.sub(r'<.+\'(.+)\'>', r'\1', str(layer_class))
    
    def select(self):
        gen_key = "gen_{}".format(self.gen_count)
        actual_gen = self.breeding_pool[gen_key]
        initial_gen_length = len(self.breeding_pool[gen_key])
        for c in range(initial_gen_length):
            if c == initial_gen_length - 1:
                break
            else:
                del self.breeding_pool[gen_key][0]
                

if __name__ == '__main__':
    # Testing EvoNet for evolution of keras model
    evonet = EvoNet(bc_features, bc_labels, start_population=5)
    # Initialize first population
    evonet.generate_initial_generation()
    # Evalutate first population and take best two for next step (mating)
    evonet.evaluate()
    print(evonet.breeding_pool["gen_1"][-1][0]["checkpoint"].best)
    # Mate best two models of previous step by selecting half-half core layers
    evonet.mate()
    print(evonet.breeding_pool)
    # Select best two models -> necassary?
    # evonet.select()
    del evonet