import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis
from pathlib import Path

path_pow = "C:/Users/pprru/Desktop/salidas2/procesado_pow.csv"
path_eda = "C:/Users/pprru/Desktop/salidas2/procesado_eda.csv"

DATA_PATH = "C:/Users/pprru/Desktop/Bueno/datos"


def read_archivo(path: str):

    print(f"Leyendo archivo: {path}")
    print(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def obtener_usuarios(path: str):
    print(f"Leyendo archivo: {path}")
    print(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if ("username" in df):
        df = df.rename(columns={"username": "user"})   # en cada base de datos el usuario se llama de una forma, se pone user para evitar errores
    users = df.pop("user")
    return users

import pandas as pd

def correlacion(entrada):
    # Cargar
    if isinstance(entrada, str):     # recibe un str
        df = read_archivo(entrada)
    elif isinstance(entrada, pd.DataFrame):   # recibe un dataframe
        df = entrada.copy()
    else:
        raise TypeError(f"entrada debe ser str o DataFrame, no {type(entrada)}")

    #  Normalizar nombres por si vienen con espacios
    df.columns = [c.strip() for c in df.columns]
    print(df.columns)

    # Separar columnas que NO deben entrar en la correlación
    cols_excluir = [c for c in ["username", "user", "epoch", "diagnosed"] if c in df.columns]

    # Features = todo menos esas columnas
    feat = df.drop(columns=cols_excluir, errors="ignore").copy()

    # 5) Correlación solo numérica
    corr = feat.corr(numeric_only=True)

    sns.heatmap(corr)

    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.show()    



    return df, corr



def eliminar_correlacion(df: pd.DataFrame, corr: pd.DataFrame, limite: float,
                         protected=("username", "user", "epoch", "diagnosed")) -> pd.DataFrame:
 

    # 1) columnas a proteger (si existen)
    protected = [c for c in protected if c in df.columns]
    df_prot = df[protected].copy()

    # 2) trabajamos SOLO con las columnas que están en corr (features)
    feat_cols = [c for c in corr.columns if c in df.columns]
    X = df[feat_cols].copy()

    # 3) máscara de selección sobre feat_cols (mismo orden que corr)
    keep = np.ones(len(feat_cols), dtype=bool)

    for i in range(len(feat_cols)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(feat_cols)):
            if keep[j] and corr.iloc[i, j] >= limite:
                keep[j] = False

    selected_feat_cols = [c for c, k in zip(feat_cols, keep) if k]

    # 4) devolver df con protegidas + features filtradas
    df_out = pd.concat([df_prot.reset_index(drop=True),
                        X[selected_feat_cols].reset_index(drop=True)], axis=1)

    print(f"[INFO] Features antes: {len(feat_cols)} | después: {len(selected_feat_cols)}")
    return df_out



if __name__ == "__main__":
    correlacion(DATA_PATH+"/bandpower_robots_all_users.csv")