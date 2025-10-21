# MNIST CNN & LSTM Classification

This project implements and compares two deep learning models — a **Convolutional Neural Network (CNN)** and an **LSTM** — on the MNIST handwritten digit dataset.  

- 🧠 **CNN Test Accuracy:** 99.09%  
- 🔁 **LSTM Test Accuracy:** 98.50%

The goal of this project was to train, tune, and evaluate two different neural network architectures using PyTorch and demonstrate the impact of model design on classification performance.

---

## 🚀 How to Run
Requirements: 
-	Python 3.10
-	PyTorch
-	torchvision
-	numpy
Steps to follow:

1: Create conda environment by:

conda create -n mnist python=3.10 -y
conda activate mnist

2: Install dependencies:

conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

3: Clone or download the project folder:

cd mnist_project

4: Run CNN

python main.py --network cnn --epochs 8 --bs 64 --lr 0.001

5: Run LSTM

python main.py --network lstm --epochs 10 --bs 128 --lr 0.001 --hidden_size 128 --num_layers 1

The MNIST dataset will be downloaded automatically on first run.
