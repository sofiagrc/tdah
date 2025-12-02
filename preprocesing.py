import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis


path_pow = "C:/Users/pprru/Desktop/salidas2/procesado_pow.csv"
path_eda = "C:/Users/pprru/Desktop/salidas2/procesado_eda.csv"

def correlacion(path: Path, tipo: str):
    print(f"Leyendo archivo: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    
    if tipo == "pow":
        df = df.iloc[:, 2:].copy()  # saltar las 2 primeras columnas
    elif tipo == "eda":
        df = df.iloc[:, 1:].copy()  # saltar la primera columna


    label_encoder = LabelEncoder()
    df.iloc[:, 0] = label_encoder.fit_transform(df.iloc[:, 0]).astype('float64')
    df.info()

    plt.figure(figsize=(10, 8))
    corr = df.corr()
    print(corr.head())
    sns.heatmap(corr)

    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.show()

    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= 0.9:
                if columns[j]:
                    columns[j] = False

    selected_columns = df.columns[columns]
    df = df[selected_columns]
    return df

def kurtosis(path: Path):
    print(f"Leyendo archivo: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    kurt = df.select_dtypes(include="number").kurtosis()
    print("Curtosis de las columnas numéricas:")
    for index,value in kurt.items():
        print(index,value)
        

    sns.kdeplot(df['FC6_beta'])
    plt.title("Datos")
    plt.xlabel("Valor")
    plt.show()

    return kurt


if __name__ == "__main__":
   
    #correlacion(path_pow,"pow")
    #correlacion(path_eda,"eda")
    #kurtosis(path_pow)
    kurtosis(path_eda)