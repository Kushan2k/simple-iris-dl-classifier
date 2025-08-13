# 🌸 Iris Deep Learning Classifier

This repository contains a deep learning-based classifier for the classic Iris flower dataset. The model is built using TensorFlow and Keras, and demonstrates how to apply neural networks to a simple multiclass classification problem.

---

## 📊 Dataset

The **Iris dataset** is a well-known dataset used for pattern recognition. It contains 150 samples of iris flowers, divided into three species: _Setosa_, _Versicolor_, and _Virginica_. Each sample has four features:

- Sepal length
- Sepal width
- Petal length
- Petal width

---

## 🧠 Model Overview

The model is a feedforward neural network (using `Sequential` from Keras) trained to classify the iris species based on the four input features. It uses:

- Dense layers with ReLU activation
- Categorical output with softmax
- Evaluation via classification report

---

## 🚀 Installation

To run this project locally, follow the steps below:

### 1. Clone the Repository

```bash
git clone https://github.com/Kushan2k/simple-iris-dl-classifier.git
cd iris-dl-classifier
```

### 2. Setup a virtual environment(Optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

#### Required Libraries:

- numpy
- matplotlib
- seaborn
- tensorflow
- scikit-learn

If requirements.txt is missing, you can install manually:

```bash
pip install numpy matplotlib seaborn tensorflow scikit-learn

```

## 🧪 How to Run

You can run the notebook using:

```bash
jupyter notebook iris_dataset.ipynb

```

Or convert to a script:

```bash
jupyter nbconvert --to script iris_dataset.ipynb
python iris_dataset.py

```

## 📈 Results

After training, the model is evaluated using:

- Accuracy
- Confusion matrix
- Classification report

You’ll also see visualizations of the training history and predictions using matplotlib and seaborn.

## 📁 File Structure

```bash
iris-dl-classifier/
│
├── iris_dataset.ipynb      # Main Jupyter notebook
├── README.md               # Project documentation
├── requirements.txt        # Dependencies (optional)
└── outputs/                # (Optional) Saved plots and model

```

## ✍️ Author

Kushan Gayantha
Software Engineer @ Freelancer

## 📄 License

This project is open source and available under the MIT License.

<i>
Let me know if you want a matching `requirements.txt` generated for this project!
</i>

Thank you