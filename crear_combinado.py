import pandas as pd
from pathlib import Path

OUT_DIR = "C:/Users/pprru/Desktop/Bueno/datos"
DATA_PATH = "C:/Users/pprru/Desktop/Bueno/datos"

path_eda = DATA_PATH+"/tabla_eda_con_diagnostico.csv"
path_pow = DATA_PATH+"/bandpower_robots_all_users.csv"


def read_archivo(path: str):
    print(f"Leyendo archivo: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


# quiero limpiar las tablas para que se queden los mismos usuarios

def obtener_usuarios_eda():
    archivo_eda = read_archivo(path= path_eda)
    etiquetas_user = archivo_eda["username"].values.tolist()
    return etiquetas_user

def obtener_usuarios_pow():
    archivo_pow = read_archivo(path = path_pow)
    etiquetas_user = archivo_pow["user"].values.tolist()
    return etiquetas_user

def validar_users():
    user_eda = obtener_usuarios_eda()
    user_pow = obtener_usuarios_pow()
    comunes = []

    for i in range(len(user_eda)):
        if ( user_eda[i] in user_pow):
            comunes.append(user_eda[i])

    return (comunes)  # contiene los usuarios comunes a las dos bases de datos


def base_datos_completa():
    # primero recorro la base de datos de pow que tiene mas filas de cada usuario
    # filtrando por los usuarios comunes

    archivo_pow = read_archivo(path_pow)
    archivo_eda = read_archivo(path_eda)

    comunes = validar_users()

    # elimino las que no sean comunes
    archivo_pow_filtrado = archivo_pow[archivo_pow["user"].isin(comunes)].copy()  # dataframe de pow con los usuarios comunes
    archivo_eda_filtrado = archivo_eda[archivo_eda["username"].isin(comunes)].copy()

    archivo_eda_filtrado = archivo_eda_filtrado.rename(columns={"username": "user"})


    df_eda_feat = archivo_eda_filtrado.set_index("user").add_prefix("EDA_")

    df_mix = archivo_pow_filtrado.merge(
        df_eda_feat,
        left_on="user",
        right_index=True,
        how="left"     # left: se quedan todas las filas de POW
    )

    return df_mix


def limpiar_tabla():
    df = base_datos_completa()
    df.drop(['EDA_diagnosed'], axis='columns', inplace=True)

    out_path = Path(OUT_DIR+"/combinada.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Tabla final combinada guardada en: {out_path}")

    y = df[["diagnosed"]]
    y_out_path = Path(OUT_DIR+"/combinada_y.csv")
    y.to_csv(y_out_path, index=False, encoding="utf-8")
    print(f"[OK] Tabla Y guardada en: {y_out_path}")

    columnas_a_quitar = [c for c in ["diagnosed", "user", "username", "epoch"] if c in df.columns]

    X = df.drop(columns=columnas_a_quitar)
    X_out_path =  Path(OUT_DIR+"/combinada_x.csv")
    X.to_csv(X_out_path, index=False, encoding="utf-8")
    print(f"[OK] Tabla X guardada en: {X_out_path}")



    


if __name__ == "__main__":
   limpiar_tabla()

