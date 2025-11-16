# Titanic Dataset - Exploratory Data Analysis (EDA)

This project performs a complete Exploratory Data Analysis (EDA) on the Titanic dataset.  
It includes data cleaning, missing value imputation, feature engineering, visualization, and summary statistics.

---

## 📁 Project Structure

```
├── eda_titanic.py         # Main EDA script
├── titanic.csv            # Dataset used in the analysis
├── figures/               # Saved plots generated during EDA
├── output/                # Summary statistics, correlations & cleaned sample
└── README.md              # Project documentation
```

---

## 🧹 Data Cleaning & Imputation (Strategy A)

- **Age → global median**  
- **Fare → median**  
- **Embarked → mode**  
- **Cabin → 'Unknown'**  
- Trim whitespace in object columns  
- Convert string `"nan"` to actual `NaN`  

---

## 🧪 Feature Engineering

- **FamilySize**  
- **IsAlone**  
- **Deck** (from Cabin)  
- **AgeGroup**  
- **Fare_log1p**  

---

## 📊 Visualizations (saved in `figures/`)

- Histograms  
- Boxplots  
- Count plots  
- Correlation heatmap  
- Survival analysis by Pclass, Sex & FamilySize  

---

## 📈 Output Files (saved in `output/`)

- `summary.csv`  
- `correlations.csv`  
- `cleaned_sample.csv`  
- `fare_outliers.csv`  

---

## ▶️ How to Run

```bash
python eda_titanic.py
```

Or specify dataset:

```bash
python eda_titanic.py titanic.csv
```

---

## 📦 Requirements

```bash
pip install pandas numpy matplotlib seaborn
```

---

## ✨ Author
Atharva Jadhav
