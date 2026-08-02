# EATC-Assignment2

FraudWatch is an end-to-end Machine Learning pipeline and web application designed to detect fraudulent credit card transactions.

- **Live Application:** [FraudWatch Streamlit App](https://hbjxxyuhzxhpzrakwk8p8o.streamlit.app/)
- **Repository:** [waltzInCSharpMajor/EATC-Assignment2](https://github.com/waltzInCSharpMajor/EATC-Assignment2)

---

## Features

- **End-to-End Pipeline:** Preprocessing, categorical encoding, feature scaling, & probabilistic inference.
- **Robust Feature Engineering:** Extracts temporal patterns (hour of day, day of week) from timestamps while removing sensitive unique IDs to prevent data leakage.
- **Imbalance Handling:** Utilizes class-weighted training (`scale_pos_weight` in XGBoost & `balanced` class weights in SVC) to detect rare fraudulent events effectively.
- **Interactive Web App:** Real-time user input & batch evaluation built with Streamlit.

---

## Data Preprocessing & Methodology

1. **Identifier Removal:** Excluded unique fields (`cc_num`, `trans_num`, `first`, `last`, `street`, `city`, `zip`, `dob`) to prevent model memorization & leakage.
2. **Temporal Extraction:** Converted raw `trans_date_trans_time` into `hour` & `day_of_week`.
3. **Categorical Encoding:** Applied `LabelEncoder` to high-cardinality features (`merchant`, `category`, `job`, `gender`) with built-in handling for unseen runtime values (`Unknown`).
4. **Feature Scaling:** Applied `StandardScaler` for distance-sensitive algorithms like Support Vector Classifiers (SVC).
5. **Stratified Splitting:** Maintained original fraud distribution across training & testing sets.

---

## Models Trained

- **Support Vector Classifier (SVC):** Trained with an RBF kernel & calibrated via `CalibratedClassifierCV` to provide confidence scores for predictions.
- **XGBoost Classifier:** Optimized gradient boosting model using positive weight scaling (`scale_pos_weight`) to optimize recall and F1-score on imbalanced target classes.


---

## Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed.

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/waltzInCSharpMajor/EATC-Assignment2.git](https://github.com/waltzInCSharpMajor/EATC-Assignment2.git)
   cd EATC-Assignment2
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Jupyter Notebook (Model Training)
   
5. Launch the Streamlit App locally:
   ```bash
   streamlit run app.py
   ```






## Acknowledgements

- Credit Card Transactions Fraud Detection Dataset (`fraudTrain.csv` and `fraudTest.csv`) by Kartik Shenoy on Kaggle (<https://www.kaggle.com/datasets/kartik2112/fraud-detection>)
  - Dataset created using Sparkov Data Generation by Brandon Harris on Github (<https://github.com/namebrandon/Sparkov_Data_Generation>)
