# Machine-Learing-Projects
![Machine-Learing-Projects](https://socialify.git.ci/XingshengXu/Machine-Learing-Projects/image?description=1&descriptionEditable=A%20collection%20of%20various%20projects%20showcase%20the%20application%20of%20machine%20learning%20algorithms%20and%20techniques%20to%20real-world%20problems.&font=Inter&logo=https%3A%2F%2Fcyberhongtu.files.wordpress.com%2F2023%2F03%2Fcyberhongtu-logo-4.png%3Fresize%3D668%252C668&name=1&pattern=Circuit%20Board&theme=Auto)

## Table of Contents
- [Introduction](#introduction)
- [Technical Requirements](#technical-requirements)
- [Installation and Setup Guide](#installation-and-setup-guide)
  - [Step 1: Clone the Repository](#step-1-clone-the-repository)
  - [Step 2: Set Up Virtual Environment](#step-2-set-up-virtual-environment)
  - [Step 3: Install Dependencies](#step-3-install-dependencies)
- [Running the Code](#running-the-code)
- [All Machine Learning Project Demos](#all-machine-learning-project-demos)

## Introduction
The "Machine Learning Projects" repository is a collection of various projects that showcase the machine learning algorithms behind each model and the application of these models to real-world problems. Each project in the repository is designed to provide a hands-on experience for individuals looking to explore the field of machine learning. The projects cover a wide range of models, from simple regression analysis to complex deep learning networks, providing a comprehensive overview of the field.

The repository is organized in a way that makes it easy for users to understand the code and the methodology used. Each project includes a main code file (.py) that contains the core algorithms for the model, written in an Object-Oriented Programming format. Additionally, a demo file (.ipynb) details the usage and results of running each model on different sets of data, and sometimes a utils file is included that contains utility functions used in the main code file. The code is well-commented and includes docstrings, making it easy to follow and suitable for individuals at all levels of experience, from beginners to advanced users.

Whether you're a data scientist, machine learning engineer, or simply interested in the field, this repository is a valuable resource for anyone looking to expand their knowledge and build their portfolio. So, dive in and start exploring the world of machine learning!

## Technical Requirements
* **Programming Language**: Python 3.11 (compatible with future versions)
* **Third-Party Libraries**: Minimal, mostly for data pre-processing and visualization. All the required libraries/packages are listed in the `requirements.txt` file.
* **External Dependencies**: None, all datasets are included within the repository.
* **Hardware Requirements**: CPU sufficient, as the models are lightweight and use small datasets for demonstration and educational purposes.

## Installation and Setup Guide
### Step 1: Clone the Repository
Clone the repository from GitHub by running the following command:
```bash
git clone https://github.com/XingshengXu/Machine-Learing-Projects.git
```
### Step 2: Set Up Virtual Environment
Navigate to the project folder and create a virtual environment using the following command:
```bash
python -m venv .venv
```
Activate the virtual environment (Optional but recommended):
* On Windows:
```bash
.venv\Scripts\activate
```
* On macOS and Linux:
```bash
source .venv/bin/activate
```
### Step 3: Install Dependencies
Navigate to the cloned repository and install the necessary packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Running the Code
Each project within this repository has its own dedicated folder, which generally includes:

* **model_name.py**: Core algorithms written in an object-oriented format.
* **model_demo.ipynb**: Demo codes to help you get started and see the model in action.
* **utils.py**: Utility functions.
* **plots**: A folder containing plots and animations.
To run a model, navigate to its folder and open the demo.ipynb Jupyter notebook. This will provide an interactive way to understand the model and its outputs.

## All Machine Learning Project Demos
### 1. Linear Regression
<p align="center">
  <img src="https://github.com/XingshengXu/Machine-Learing-Projects/assets/125934684/afe82484-84d8-4895-b68f-e41fc6d7bc7d.gif" width="600">
  <br>
  <em>Linear Regression (BGD) of Synthetic Data Set</em>
</p>
<p align="center">
  <img src="https://github.com/XingshengXu/Machine-Learing-Projects/assets/125934684/1878e323-c58b-4821-8c34-bbe0d8ac7c05.gif" width="600">
  <br>
  <em>Linear Regression (SGD) of Synthetic Data Set</em>
</p>

### 2. Logistic Regression
<p align="center">
  <img src="https://github.com/XingshengXu/Machine-Learing-Projects/assets/125934684/89f7513b-7693-48db-9b00-eb3e30822893" width="600">
  <br>
  <em>Training Data with Decision Boundary of Non-Linearly Separable Synthetic Data (BGD)</em>
</p>

### 3. Neural Perceptron
<p align="center">
  <img src="https://github.com/XingshengXu/Machine-Learing-Projects/assets/125934684/9f0fc0ea-1d16-4f1e-9dd2-b55b75bb42ae" width="600">
  <br>
  <em>Decision Boundaries Visualized by Multi-Layer Perceptron on Synthetic Data</em>
<p>

### 4. Radial Basis Function Networks
<p align="center">
  <img src="https://github.com/XingshengXu/Machine-Learing-Projects/assets/125934684/dfdc7ba7-fdb4-4966-a7a1-1e4a31f35b1a" width="600">
  <br>
  <em>Decision Boundaries Visualized by RBF Networks on Synthetic Data</em>
<p>
<p align="center">
  <img src="https://github.com/XingshengXu/Machine-Learing-Projects/assets/125934684/154b6f41-82c5-481d-8ee4-ce038862e91d" width="600">
  <br>
  <em>Implement Radial Basis Function Networks Regressor with Different Cluster Numbers on Nonlinear Data</em>
<p>

### 5. Hopfield Neural Network
<p align="center">
  <img src="https://github.com/XingshengXu/Machine-Learing-Projects/assets/125934684/157f2268-abc7-4537-9a8f-85eb7152e0d3" width="400">
  <br>
  <em>Image Noise Reduction using Hopfield Neural Network (digit 0)</em>
<p>
<p align="center">
  <img src="https://github.com/XingshengXu/Machine-Learing-Projects/assets/125934684/fb631b43-354b-4cb2-9f37-ce0fab639f55" width="400">
  <br>
  <em>Image Noise Reduction using Hopfield Neural Network (digit 1)</em>
<p>

### 6. Adaptive Resonance Theory
<p align="center">
  <img src="https://github.com/XingshengXu/Machine-Learing-Projects/assets/125934684/2de6fe67-436e-4aa8-8973-167e3d443027" width="600">
  <br>
  <em>Decision Boundaries Visualized by Fuzzy Adaptive Resonance Theory Classifier</em>
<p>
<p align="center">
  <img src="https://github.com/XingshengXu/Machine-Learing-Projects/assets/125934684/bd07ab4b-06ea-467a-b68d-a667fb5a8574" width="600">
  <br>
  <em>Fuzzy Adaptive Resonance Theory Based Image Compression</em>
<p>

### 7. Generative Learning Algorithms
<p align="center">
  <img src="https://github.com/XingshengXu/Machine-Learing-Projects/assets/125934684/35888f6f-2cf4-4382-9e3f-5481c03228c5" width="600">
  <br>
  <em>Gaussian Discriminant Analysis for Classification Marketing Target</em>
<p>

### 8. Support Vector Machine
### 9. K-Nearest Neighbors
### 10. Decision Tree
### 11. Ensemble Learning
### 12. Principal Component Analysis
### 13. Locally Linear Embedding
### 14. Maximum Variance Unfolding
### 15. Independent Component Analysis


