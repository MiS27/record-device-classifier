# Dataset description
Dataset is not available in repository.
Dataset contained two sets of wave files: A and B. Each set contained 500 files numbered from 1 to 500.
Each file was a 4-second long recording. Files in each set were recorded using different device, in different conditions.

# Task
Develop a machine learning algorithm that, given an unknown 4-second recording, decides whether it comes from set A or B.

# Preprocessing
Program preprocess data by generating spectrogram for each WAV file. Data in this form is used in 5-fold train-and-test procedure.

# Cross-validation
In each fold processing the non-test data is further splitted into training (75% of non-test data) and validation (25%) sets. The learning process consists of 100 epochs. In each epoch model is being train using 12 minibatches consisting of 50 examples each. After each epoch model is being tested on validation set to estimate perfomance for unkown data and avoid overfitting by performing early stopping, which is triggered if highest accuracy on validation data was obtain more than 20 epochs ago.

# Network
As learning model a neural network implemented with Lasagne framework was used. The inputs of the network are spectrograms of WAV files. The network is build with 5 trainable layer: 2 Convolutional, 1 GRU recurrent layer and 2 Dense layers. At the output of the network we receive single value from 0 to 1. Which is the probablilty that processed record comes from set A. 

|Nr |Type  |Output shape | Kernel size (units) / stride |
|---|-----|------|----------------------|
|0 |Input  |129 x 35 |  |
|1 |Convolution |16 x 129 x 35 | 3x3 / 1x1 |
|2 |Convolution |16 x 129 x 35 | 3x3 / 1x1 |
|3 |Dimshuffle |35 x 16 x 129||
|4 |Flatten |35 x 2064||
|5 |GRU   |16 | 16 |
|6 |Dense  |16 | 16 |
|7 |Dense  |1 | 1 |

# Results
Final score in the accuracy of classification of test recordings across all folds was 88.5%.
