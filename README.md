# EvoNet
## Description 
This is an approach of creating a genetic algorithm with keras for evolving very suitable neural networks for a specific dataset or types of datasets with similar feature and label dimensions/shapes.
<br/>
* __Initializing first population__: Generating RandNets (keras functional api) of a given count.
* __Evaluation__: Select best two networks for mating.
* __Mating__: Mate the core layers of the selected individuals randomly. (Crossover not yet implemented)
* __Mutate__: Insert mutations -> Alterations in layer hyperparameters. (Not yet implemented)

## Goal
There are a lot of optimization algorithms for neural networks out there, but as I have a life science background I decided to use a genetic algorithm to optimize the neural network architectures. If you have a ready neural network for example, it is only suitable for the dataset it was developed for. So if you have a new dataset you will propably have to change or adjust your neural network architecture or even create a completly new one.

## Contribution
If you also like this idea, please come and support us here. We are very happy about a nice contribution :pray:
