# 💳 Credit Card Fraud Detection

This project implements an end-to-end Credit Card Fraud Detection system using a real-world dataset. It encompasses data preprocessing, model training, evaluation, and deployment through an interactive dashboard.

---

## 📌 Project Overview

Credit card fraud poses significant challenges to financial institutions, leading to substantial monetary losses annually. This project aims to detect fraudulent transactions using machine learning techniques, providing an efficient tool to mitigate such risks.

---

## 🧰 Features

- **Data Preprocessing**: Handles missing values, feature scaling, and encoding.
- **Model Training**: Utilizes algorithms like Logistic Regression, Decision Trees, and Random Forests.
- **Model Evaluation**: Assesses models using metrics such as accuracy, precision, recall, and F1-score.
- **Hyperparameter Tuning**: Optimizes model performance through grid search.
- **Interactive Dashboard**: Built with Dash to visualize data insights and model predictions.

---

<pre> 
## 📂 Project Structure 

``` 
project_4/ 
├── CreditCardFraudDetection_jpynb1.ipynb       # Jupyter Notebook with EDA and model development 
├── app.py                                      # Dash application script 
├── creditcard.db                               # SQLite database containing transaction data 
├── creditcard_sample.csv                       # Sample dataset for quick testing 
├── final_model.pkl                             # Serialized trained model 
├── model_optimization_results.csv              # Results from hyperparameter tuning 
├── requirements.txt                            # Python dependencies 
└── README.md                                   # Project documentation 
``` 
</pre>
---

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/estortega/project_4.git
   cd project_4

2. **Install dependencies**:
pip install -r requirements.txt

## 🧪 Usage
1. **Run the Dash application**
python app.py

2. **Access the dashboard:**
Open your web browser and navigate to https://project-4-jzx5.onrender.com/ to interact with the application.

## 📊 Dataset

The dataset used is a subset of the Kaggle Credit Card Fraud Detection dataset, which contains transactions made by European cardholders in September 2013.
## 📈 Model Performance

The final model achieved the following performance metrics on the test set:

Accuracy: 0.9995 

Precision:  Likely very high (especially due to class balancing and low variance), though not directly in this CSV

Recall:  Likely very high (especially due to class balancing and low variance), though not directly in this CSV

## 📌 Future Improvements
Implement real-time data streaming for live fraud detection.

Integrate with cloud-based databases for scalability.

Enhance the dashboard with more interactive features and visualizations.

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 🙌 Acknowledgments
Kaggle for providing the dataset.

Dash for the interactive dashboard framework.

Contributers:
Rache Morris, Esteban Ortega, Haby Sarr, Krishna Sigdel