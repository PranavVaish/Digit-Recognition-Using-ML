# 🧠 Handwritten Digit Recognition using ML (MNIST + HOG)

This project implements a classic **Handwritten Digit Recognition** system using handcrafted features and traditional machine learning models — **K-Nearest Neighbours (KNN)**, **Support Vector Machine (SVM)**, and **Random Forest (RF)** — evaluated on the **MNIST dataset**.  
It uses **Histogram of Oriented Gradients (HOG)** for feature extraction and avoids deep learning to emphasize transparency and educational insight.

---

## 📌 Objective

- Recognize digits (0–9) from MNIST using **non-deep-learning** models.
- Compare the performance of KNN, SVM, and Random Forest.
- Understand how classical ML and handcrafted features perform vs deep learning.
- Train models on **HOG descriptors** extracted from grayscale images.

---

## 🚀 Project Structure

```
Digit-Recognition-Using-ML/
│
├── data/                    # Contains MNIST dataset (downloaded via sklearn)
├── src/
│   ├── feature_extraction.py   # Uses HOG to extract features
│   ├── train_knn.py            # KNN model training + evaluation
│   ├── train_svm.py            # SVM model training + evaluation
│   ├── train_rf.py             # Random Forest training + evaluation
│   └── visualize.py            # Feature visualization & confusion matrices
│
├── results/                # Confusion matrices and accuracy results
├── README.md               # You are here
└── requirements.txt        # Python dependencies
```

---

## 🧮 Workflow

1. **Data Preprocessing**  
   Load and normalize MNIST grayscale images (28x28).

2. **Feature Extraction: HOG**
   - Convert each digit image into a **144-length HOG feature vector**.
   - This captures shape, edge, and stroke direction details.

3. **Model Training**
   - KNN: Varies 'k', uses Euclidean distance
   - SVM: Kernel = linear, RBF with tuning
   - Random Forest: 100 estimators with tuned depth and splits

4. **Evaluation**
   - Accuracy
   - Confusion Matrix
   - Cross-comparison

---

## 🔍 Comparative Performance

| Model        | Accuracy (%) |
|--------------|--------------|
| KNN          | 93.38%       |
| SVM          | 94.78% ✅     |
| Random Forest| 92.45%       |

✅ SVM performed best in terms of generalization and precision.

---

## 🛠️ Installation (Windows/Linux/Mac)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/PranavVaish/Digit-Recognition-Using-ML.git
cd Digit-Recognition-Using-ML
```

### 2️⃣ Create Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Dependencies

- numpy
- matplotlib
- scikit-learn
- scipy
- joblib

_Install with_ `pip install -r requirements.txt`

---

## 📈 How to Run

### ➤ Train all models:

```bash
python src/train_knn.py
python src/train_svm.py
python src/train_rf.py
```

Each script will:
- Train its respective model
- Evaluate using test set
- Plot confusion matrix
- Print accuracy score

---

## 📊 Visualizations

- HOG Feature Maps
- Confusion Matrices
- Model Accuracy Comparison
- Feature Importance (Random Forest)

Images are auto-saved in the `results/` folder after each run.

---

## 📚 Educational Insight

This project is ideal for:
- Learning classical ML pipelines
- Understanding HOG feature engineering
- Comparing traditional ML vs deep learning tradeoffs
- Efficient digit recognition without GPU or neural nets

---

## 🤝 Contributors

- [Pranav Vaish](https://www.linkedin.com/in/pranavvaish20/)
- Kartik Goel  
- Anish Mahajan  

Supervised by: **Ms. Kanika**

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🌐 Acknowledgements

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Dalal and Triggs](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) for HOG
- [scikit-learn](https://scikit-learn.org/stable/) for models & utilities

---

> 💡 _"Old is gold — Traditional ML still rocks in the right problem space."_  
> — Team UCS411
