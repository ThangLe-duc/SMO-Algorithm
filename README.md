# SMO-Algorithm
We contribute here a Matlab-based package for training deep neural networks (DNN)s based on our novel optimization framework named Sequential Motion Optimization (SMO). We select the popular MNIST classification problem to demo in this package. Some detailed definitions and instructions are also presented into the codes to help the users easily modify it for personal research and uses. We show below some necessary information of the SMO package including
1.	Structure of SMO software package:
• Main.m: includes the main codes for running SMO to solve DNN learning problems.
• Folder “SMO Package”: includes the functions defining the SMO procedures to optimize DNN parameters.
• Folder “Datasets”: includes the problem datasets.
2.	How to solve DNN training problems by SMO software package:
• Put the problem dataset into the “Datasets” folder and specify the problem in the function ClassificationProblems.m
• Define the network structures, general learning hyperparameters and SMO settings in the main function Main.m via the functions DNNStructure.m, DNN_Setup.m, DNN_LearningRate.m, SMO_Setup.m, Batch_Setup.m and RankSetup.m
• Run Main.m and wait until the training process stops.
• The network parameters and results are recorded in the DNN.mat and results.mat, respectively.
# Contributors
Thang Le-Duc

Quoc-Hung Nguyen

Jaehong Lee

H. Nguyen-Xuan

# Reference
Le-Duc, T., Nguyen, Q. H., Lee, J., & Nguyen-Xuan, H. (2022). Strengthening Gradient Descent by Sequential Motion Optimization for Deep Neural Networks. IEEE Transactions on Evolutionary Computation. https://ieeexplore.ieee.org/abstract/document/9766043
