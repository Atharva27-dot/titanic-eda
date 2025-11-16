import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Utility functions

def find_dataset_path(arg_path=None):
    candidates = []
    if arg_path:
        candidates.append(Path(arg_path))
    # common names
    candidates += [Path("data/titanic.csv"),
                   Path("data/train.csv"),
                   Path("titanic.csv"),
                   Path("train.csv")]
    for p in candidates:
        if p.exists():
            return p
    return None

def ensure_dirs():
    Path("figures").mkdir(parents=True, exist_ok=True)
    Path("output").mkdir(parents=True, exist_ok=True)

def save_fig(fig, name):
    fname = Path("figures") / name
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


# EDA steps as functions

def load_data(path):
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    return df

def quick_checks(df):
    print("\n--- Quick checks ---")
    print(df.info())
    print("\nTop 5 rows:")
    print(df.head().to_string(index=False))
    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False))

def initial_cleaning_strategy_A(df):
    """
    Imputation Strategy A:
      - Age -> global median
      - Embarked -> mode
      - Fare -> median
      - Cabin -> 'Unknown' (if present)
      - Trim whitespace in object columns
    """
    df = df.copy()

    # Trim whitespace in object columns and normalize 'nan' strings
    obj_cols = df.select_dtypes(include=['object']).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip().replace({'nan': np.nan})

    # Embarked -> mode
    if 'Embarked' in df.columns:
        mode_emb = df['Embarked'].mode()
        if len(mode_emb) > 0:
            df['Embarked'] = df['Embarked'].fillna(mode_emb.iloc[0])
        else:
            df['Embarked'] = df['Embarked'].fillna('S')  # fallback

    # Fare -> median
    if 'Fare' in df.columns:
        fare_median = df['Fare'].median()
        df['Fare'] = df['Fare'].fillna(fare_median)

    # Age -> global median
    if 'Age' in df.columns:
        age_median = df['Age'].median()
        df['Age'] = df['Age'].fillna(age_median)

    # Cabin -> Unknown
    if 'Cabin' in df.columns:
        df['Cabin'] = df['Cabin'].fillna('Unknown')

    return df

def feature_engineering(df):
    df = df.copy()
    # Family size
    if set(['SibSp','Parch']).issubset(df.columns):
        df['FamilySize'] = df['SibSp'].fillna(0).astype(int) + df['Parch'].fillna(0).astype(int) + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    # Deck from Cabin (first letter)
    if 'Cabin' in df.columns:
        df['Deck'] = df['Cabin'].astype(str).str[0].replace({'U': 'Unknown'})
    # Age bins
    if 'Age' in df.columns:
        bins = [0, 12, 20, 40, 60, 120]
        labels = ['Child', 'Teen', 'Adult', 'MiddleAge', 'Senior']
        df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
    # Log transform Fare (new column) to reduce skew for plots
    if 'Fare' in df.columns:
        df['Fare_log1p'] = np.log1p(df['Fare'])
    return df

def numeric_summary(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = df[num_cols].describe().T
    # add skew
    summary['skew'] = df[num_cols].skew()
    return summary

def plot_univariate_numeric(df, cols=None):
    if cols is None:
        cols = ['Age','Fare']
    for c in cols:
        if c not in df.columns:
            continue
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        sns.histplot(df[c].dropna(), kde=True, ax=ax[0])
        ax[0].set_title(f'{c} distribution')
        sns.boxplot(x=df[c].dropna(), ax=ax[1])
        ax[1].set_title(f'{c} boxplot')
        save_fig(fig, f'univariate_{c}.png')

def plot_categorical_counts(df, cols=None, hue=None):
    if cols is None:
        cols = ['Pclass','Sex','Embarked','Survived']
    for c in cols:
        if c not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6,4))
        if hue and hue in df.columns:
            sns.countplot(data=df, x=c, hue=hue, ax=ax)
        else:
            sns.countplot(data=df, x=c, ax=ax)
        ax.set_title(f'Count of {c}')
        save_fig(fig, f'count_{c}.png')

def correlation_and_heatmap(df):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        print("Not enough numeric columns for correlation.")
        return None
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title('Correlation matrix')
    save_fig(fig, 'correlation_matrix.png')
    return corr

def bivariate_plots(df):
    # Survival by Pclass & Sex
    if set(['Pclass','Sex','Survived']).issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(data=df, x='Pclass', y='Survived', hue='Sex', ci=None, ax=ax)
        ax.set_title('Survival rate by Pclass and Sex')
        save_fig(fig, 'survival_pclass_sex.png')
    # FamilySize vs Survival
    if 'FamilySize' in df.columns and 'Survived' in df.columns:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.pointplot(data=df, x='FamilySize', y='Survived', ax=ax)
        ax.set_title('Survival rate by FamilySize')
        save_fig(fig, 'survival_familysize.png')


# Main pipeline

def main(argv):
    ensure_dirs()
    arg_path = argv[1] if len(argv) > 1 else None
    path = find_dataset_path(arg_path)
    if path is None:
        print("ERROR: Could not find dataset. Try: python eda_titanic.py path/to/titanic.csv")
        print("Searched common names: data/titanic.csv, data/train.csv, titanic.csv, train.csv")
        return

    df = load_data(path)
    quick_checks(df)

    # Apply Imputation Strategy A
    df_clean = initial_cleaning_strategy_A(df)
    df_feat = feature_engineering(df_clean)

    print("\n--- Numeric summary (saved to output/summary.csv) ---")
    summary = numeric_summary(df_feat)
    print(summary.head().to_string())
    summary.to_csv("output/summary.csv")

    print("\n--- Missing values after cleaning ---")
    print(df_feat.isnull().sum().sort_values(ascending=False))

    # Save a sample of cleaned data (first 200 rows) for submission
    df_feat.head(200).to_csv("output/cleaned_sample.csv", index=False)

    # Plots
    print("\nGenerating plots in ./figures ...")
    plot_univariate_numeric(df_feat, cols=['Age','Fare','Fare_log1p'])
    plot_categorical_counts(df_feat, cols=['Pclass','Sex','Embarked','Survived'])
    corr = correlation_and_heatmap(df_feat)
    if corr is not None:
        corr.to_csv("output/correlations.csv")
    bivariate_plots(df_feat)

    # Groupby examples & prints
    if set(['Pclass','Survived']).issubset(df_feat.columns):
        print("\nSurvival rate by Pclass:")
        print(df_feat.groupby('Pclass')['Survived'].mean().round(3))

    if set(['Sex','Survived']).issubset(df_feat.columns):
        print("\nSurvival rate by Sex:")
        print(df_feat.groupby('Sex')['Survived'].mean().round(3))

    if set(['Pclass','Sex','Survived']).issubset(df_feat.columns):
        print("\nSurvival rate by Pclass and Sex:")
        print(df_feat.groupby(['Pclass','Sex'])['Survived'].mean().unstack().round(3))

    # Outlier detection example for Fare
    if 'Fare' in df_feat.columns:
        Q1 = df_feat['Fare'].quantile(0.25)
        Q3 = df_feat['Fare'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df_feat[(df_feat['Fare'] < Q1 - 1.5*IQR) | (df_feat['Fare'] > Q3 + 1.5*IQR)]
        print(f"\nFare outliers detected: {len(outliers)} rows (example saved to output/fare_outliers.csv)")
        outliers.head(50).to_csv("output/fare_outliers.csv", index=False)

    print("\nEDA complete. Key files:")
    for p in Path("figures").iterdir():
        print(" -", p)
    for p in Path("output").iterdir():
        print(" -", p)

if __name__ == "__main__":
    main(sys.argv)
