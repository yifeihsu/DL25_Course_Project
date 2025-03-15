# DL25 Course Project

This repository contains the code for the DL25 Course Project 1.

## Repository Structure

The repository includes the following Python scripts:

- `main.py`: The main script to run the training of SEResNet-CIFAR model.
- `model.py`: Defines the SEResNet model architecture.
- `continue_train.py`: (Optional) Script to continue training from a checkpoint.​
- `evaluate_testset.py`: Evaluates the model on the test dataset.
- `compare_submission.py`: Compares different submissions or model outputs.
- `demo_notebook.ipynb`: A demo notebook for defining the network, data augmentation, and training or testing. Visualizations are also included. The test loss/accuracy figure is included separately in draw_fig.py.
- `draw_fig.py`: Plotting the testing accuracy and loss for the 900 epochs.
## Getting Started
To reproduce the leaderboard results, first execute `main.py` for **750 epochs**, followed by `continue_train.py` for approximately **500 additional epochs**. The final test set accuracy should reach around **97.80%**.

For a brief demonstration, please refer to [`demo_notebook.ipynb`](demo_notebook.ipynb).
## Acknowledgement
We would like to thank the generous support and computational resources provided by the NYU High Performance Computing (HPC) center and the NYU IT team. Their infrastructure and technical assistance were instrumental in training and evaluating our SEResNet models on the CIFAR-10 dataset.

We also acknowledge the valuable help of Large Language Models (LLMs), such as ChatGPT and DeepSeek, for enhancing the code readability, suggesting improvements in our software framework, and assisting with experiment design. Their guidance significantly streamlined our workflow and implementation.