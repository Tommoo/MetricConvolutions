# About the code


**Disclaimer**

Our paper is first and foremost a theoretical work. We here provide some of our code that we used to illustrat the "metric convolution" operators using explicit rather than implicit metric. We did not have any form of efficiency in mind when writing it. As such, the code is not written in any optimised way, especially for the demo uses (single image playground and denoising). For more factorised and modular code, see the experiments on classification datasets and neural networks.


Significant room for time and memory improvements exist. You are most welcome to reimplement yourself in a very efficient manner some form of metric convolution.

---

The folders correspond to experimental settings described in the paper.

- single_image_playground: This folder contains basic playground code to get familiar with the different components of metric convolutions. 

- naive_denoising: This folder contains basic code for getting familiar with learning on a dataset with metric convolutions.

- classification_deep_networks: This folder contains the code to incorporate metric convolutions within neural networks. It is our reference (and more modular code).