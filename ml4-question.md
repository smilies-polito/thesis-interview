## Objective
The goal of this exercise is to evaluate your understanding of the basic components of a deep learning workflow using **PyTorch**.

You will implement a simple neural network that classifies handwritten digits from the **MNIST dataset**.

The execise focuses on the following aspects:
- loading and handling datasets
- defining a neural network model in Pytorch
- implementing a training loop
- evaluating model performance on a test set 

---

# Dataset

You must use the **MNIST dataset**, available through `torchvision.datasets`.

Implementation details:

- size: **28 × 28**
- grayscale (**1 channel**)
- label: **digit from 0 to 9**
- batch size: 64  

--- 

# Model 

You must implement a fully connected neural network with the following structure:
First define the in\_features for the two layers and the out\_features for Layer2 ("_" in the layer definition). 

- Linear Layer ( _ , 128)
- ReLu activation
- Linear Layer ( _ , _)
- No softmax

---

# Training the Model

You must implement a training loop with the following requirements:

- CrossEntropy Loss
- Adam Optimizer 
- learning rate = 1e-3
- 100 epochs

---

# Evaluate the Model

Evaluate the model on the test dataset. Compute overall accuracy 

---

# Optionals

- Use GPU acceleration


