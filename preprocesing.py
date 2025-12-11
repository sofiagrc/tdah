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

def correlacion(entrada):
    print(f"Es un: {type(entrada)}")
    print(entrada)

    if type(entrada)==str:
        data = read_archivo(entrada)
        print("entra aqui")

    elif type(entrada)==pd.DataFrame:
        data = entrada


    if ("username" in data):
        data = data.rename(columns={"username": "user"}) 
        users = data.pop("user")

    if ("epoch" in data):
        epoca = data.pop("epoch")

    if ("diagnosed" in data):
        epoca = data.pop("diagnosed")

    data = data.iloc[:,1:]  #coge todas las filas
    label_encoder = LabelEncoder()
    data.iloc[:,0]= label_encoder.fit_transform(data.iloc[:,0]).astype('float64') # revisar
    data.info()
    
    plt.figure(figsize=(10, 8))   # opcional, solo para que se vea más grande
    corr = data.corr()
    print(corr.head())
    sns.heatmap(corr)

    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.show()    

    
    return data, corr


def eliminar_correlacion(data,corr, limite):

    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= limite:
                if columns[j]:
                    columns[j] = False    

    selected_columns = data.columns[columns]
    print(selected_columns.shape)
    data = data[selected_columns]
    
    # las que siguen siendo verdderas y se guardan son las que no tienen correlacion
    print(data)
    return data


if __name__ == "__main__":
    correlacion(path_eda,"eda")