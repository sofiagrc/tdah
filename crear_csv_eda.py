#%% ----------------------------------------------------------------------------------------
# Rutas básicas de los archivos del dataset

from pathlib import Path
from typing import List, Tuple

import re
import pandas as pd
import numpy as np
import mne
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from mne_features.univariate import compute_pow_freq_bands
from limpieza import get_eligible_users  
import statsmodels.formula.api as sm
from sklearn.impute import SimpleImputer

BASE_DIR   = Path(r"C:\Users\pprru\Desktop\Balladeer\balladeer_data")                 # raíz UB####
EDA_CSV    = Path(r"C:\Users\pprru\Desktop\Balladeer\balladeer_embraceplus_data.csv") # EDA/embrace
DEMOG_JSON = Path(r"C:\Users\pprru\Desktop\Balladeer\users_demographics.json")
OUT_DIR    = Path(r"C:\Users\pprru\Desktop\salidas2")                                 # salidas
OUT_DIR.mkdir(parents=True, exist_ok=True)


#%% ----------------------------------------------------------------------------------------
# Rutas básicas de los archivos del dataset

from pathlib import Path
from typing import List, Tuple

import re
import pandas as pd
import numpy as np
import mne

from mne_features.univariate import compute_pow_freq_bands
from limpieza import get_eligible_users  # usa tu módulo existente

BASE_DIR   = Path(r"C:\Users\pprru\Desktop\Balladeer\balladeer_data")                 # raíz UB####
EDA_CSV    = Path(r"C:\Users\pprru\Desktop\Balladeer\balladeer_embraceplus_data.csv") # EDA/embrace
DEMOG_JSON = Path(r"C:\Users\pprru\Desktop\Balladeer\users_demographics.json")
OUT_DIR    = Path(r"C:\Users\pprru\Desktop\salidas2")                                 # salidas
OUT_DIR.mkdir(parents=True, exist_ok=True)




#%% Nombre actividades 
def es_columna_valida(serie: pd.Series) -> bool:  # -----
    
    s = serie.dropna() # se quitan los NaN
    if s.empty:  
        return False

    # se cogen filas de la columna
    muestra = s.head(20).astype(str)

    # Si parece lista/array en texto, la descartamos
    if muestra.str.contains(r"\[").any():
        return False

    # Intentamos convertir a numérico
    convertida = pd.to_numeric(muestra, errors="coerce")
    return convertida.notna().all()



def sacar_columna_media_actividad(actividad2: str, path: Path = EDA_CSV) -> List[str]:  # ------
    dataframe = pd.read_csv(path, sep=";")
    dataframe.columns = [c.strip() for c in dataframe.columns]

    columnas_keep: List[str] = []
    contexto = "Robots"

    for col in dataframe.columns:
        if contexto in col and "mean" in col and "two" not in col and actividad2 in col and  "middle" not in col:
            columnas_keep.append(col)

    return columnas_keep


def abrir_archivo_eda(path: Path = EDA_CSV, contexto: str = "Robots") -> pd.DataFrame: # -----
    
    # Leer CSV (con el separador ;)
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]  # limpiar espacios

    columnas_keep: List[str] = []

    for col in df.columns:
        # Filtramos columnas
        if contexto in col and not "mean" in col and not "missing" in col and not "count" in col: 
            # No contiene ni mean ni missing ni count
            if es_columna_valida(df[col]):
                columnas_keep.append(col)

    if "username" in df.columns:
        columnas_keep.insert(0, "username")

    if not columnas_keep:
        print(f"[WARN] No se han encontrado columnas escalares para contexto='{contexto}'.")
        return pd.DataFrame()  # DataFrame vacío

    df_sel = df[columnas_keep].copy()

    # Guardar tabla en OUT_DIR
    path_tabla = OUT_DIR / "tabla_eda1.csv"
    df_sel.to_csv(path_tabla, index=False, encoding="utf-8")
    #print(f"[OK] Tabla EDA guardada en: {path_tabla}")

    return df_sel


eda_actividades = [
    "eda_values",
    "prv_values_ms",
    "pulse_rate_values_bpm",
    "temperature_rate_values_brpm",
    "respiratory_rate_values_brpm",
    "wearing_detection_values_percentage",
]
    

def medidas_validas() -> pd.DataFrame:     
    

    medidas_eda = 0
    medidas_prv = 0
    medidas_pulse = 0
    medidas_temperature = 0
    medidas_respiratory = 0
    medidas_wearing = 0

    # Cargar tabla filtrada (sin mean/missing/count)
    df_robots = abrir_archivo_eda()

    # Cargar CSV original (con todas las columnas, incluidas las mean)
    df_raw = pd.read_csv(EDA_CSV, sep=";")
    df_raw.columns = [c.strip() for c in df_raw.columns]

    # Cargar tabla
    df_robots = abrir_archivo_eda()
    nombre_columnas = df_robots.columns.to_list()

    # Contar cuántas columnas hay de cada actividad
    for col in nombre_columnas:
        if "eda" in col:
            medidas_eda += 1
        elif "prv" in col:
            medidas_prv += 1
        elif "pulse" in col:
            medidas_pulse += 1
        elif "temperature" in col:
            medidas_temperature += 1
        elif "respiratory" in col:
            medidas_respiratory += 1
        elif "wearing" in col:
            medidas_wearing += 1

    lista_medidas = [
        medidas_eda,
        medidas_prv,
        medidas_pulse,
        medidas_temperature,
        medidas_respiratory,
        medidas_wearing,
    ]

    lista_nomb = [
        "eda",
        "prv",
        "pulse",
        "temperature",
        "respiratory",
        "wearing",
    ]

    tabla_mod = df_robots.copy()

    # Rellenar hasta 9 medidas por actividad
    for idx, n_medidas in enumerate(lista_medidas):
        if n_medidas < 9:
            diferencia = 9 - n_medidas

            # columnas 'mean' de esa actividad (lista de nombres)
            columnas_mean = sacar_columna_media_actividad(lista_nomb[idx])

        
            col_base = columnas_mean[0]
            

            # añadir columnas que faltan
            for k in range(diferencia):
                nueva_col = f"{col_base}_{9-2-k}"   # hay que ponerle el nombre de la columna correcta
                # insertar antes de las dos últimas columnas
                loc = (9*idx)+n_medidas   # 9 medidas de cada uno, añadirla justo por donde vaya midiendo (n_medidas)
                print(len(tabla_mod.columns))
                if col_base not in df_raw.columns:
                    print(f"[WARN] '{col_base}' no está en df_raw, se salta esta actividad.")
                    continue
                if col_base in df_raw.columns:     #ajustar etiquetas de los nombres de las columnas para añadir en medio 
                    # nombre antiguo (en la posición loc de tabla_mod)
                    vieja = tabla_mod.columns[loc]

                    # Construimos un nuevo nombre a partir de 'vieja'
                    partes = vieja.split("_")
                    if partes[-1].isdigit():
                        # si termina en número, lo incrementamos
                        nuevo_idx = int(partes[-1]) + 1
                        nueva = "_".join(partes[:-1] + [str(nuevo_idx)])
                    else:
                        # si no tiene número al final, le ponemos _1
                        nueva = vieja + "_1"

                    # Renombrar en el DataFrame
                    tabla_mod.rename(columns={vieja: nueva}, inplace=True)


                tabla_mod.insert(loc, nueva_col, df_raw[col_base])
                print(f" Se ha insertado una columna en {col_base}")
                print(len(tabla_mod.columns))


    path_tabla2 = OUT_DIR / "tabla_eda2.csv"
    tabla_mod.to_csv(path_tabla2, index=False, encoding="utf-8")
    #print(f"[OK] Tabla EDA guardada en: {path_tabla2}")

    return tabla_mod

"""""
    # Borrar las primeras y últimas columnas de cada actividad
    tabla_mod = df_robots.copy()

    for i in range(len(eda_actividades)):
        n = lista_medidas[i]

        tabla_mod.drop(
            columns=[f"Robots_{eda_actividades[i]}_0"],
            inplace=True,  # se pone inplace= True porque al hacer drop se devuelve un nuevo DataFrame
            errors="ignore",
        )
        tabla_mod.drop(
            columns=[f"Robots_{eda_actividades[i]}_1"],
            inplace=True,
            errors="ignore",
        )
        tabla_mod.drop(
            columns=[f"Robots_{eda_actividades[i]}_{n-1}"],
            inplace=True,
            errors="ignore",
        )
        tabla_mod.drop(
            columns=[f"Robots_{eda_actividades[i]}_{n-2}"],
            inplace=True,
            errors="ignore",
        )

    return tabla_mod

"""""



def invalidar_actividades_nulos(df: pd.DataFrame, max_nulos_por_actividad: int = 2) -> pd.DataFrame:
    
    # Se recorren todas las actividades de todas las filas, si en las columnas de cada actividad hay mas de dos null, entonces esa actividad ya no sirve

    actividades = ["eda", "prv", "pulse", "temperature", "respiratory", "wearing"]

    # Trabajamos sobre una copia para no modificar el df original sin querer
    df_mod = df.copy()

    for act in actividades:

        cols_act = [c for c in df_mod.columns if act in c]  # saca las columnas de cada actividad

        if not cols_act:
            print(f"[AVISO] No hay columnas para la actividad '{act}'")
            continue

        #print(f"[INFO] Actividad '{act}' usa columnas: {cols_act}")


        n_nulos = df_mod[cols_act].isna().sum(axis=1)   # cuenta nulos que hay en las columnas de la actividad


        filas_malas = n_nulos > max_nulos_por_actividad # si hay mas nulos que el max establecido (2) en esa fila la actividad no es válida

        print(f"[INFO] Filas con actividad '{act}' NO válida: {filas_malas.sum()}")

        df_mod.loc[filas_malas, cols_act] = np.nan  # borra toda la actividad y la pone a NaN


    
    path_tabla3 = OUT_DIR / "tabla_eda3.csv"
    df_mod.to_csv(path_tabla3, index=False, encoding="utf-8")
    #print(f"[OK] Tabla EDA guardada en: {path_tabla3}")

    return df_mod




def filtrar_actividades(df3: pd.DataFrame)-> pd.DataFrame:

    actividades = ["eda", "prv", "pulse", "temperature", "respiratory", "wearing"]
    tabla_filtrada = df3.copy()

    for act in actividades:
        # actividades a eliminar: respiratory, prv, wearing 
        if act in ("prv", "respiratory", "wearing"):
            # columnas que pertenecen a esa actividad
            cols_act = [c for c in tabla_filtrada.columns if act in c]

            print(f"[INFO] Eliminando columnas de '{act}': {cols_act}")
            tabla_filtrada.drop(columns=cols_act, inplace=True)   # se eliminan todas las columnas de la actividad

    path_tabla4 = OUT_DIR / "tabla_eda4.csv"
    tabla_filtrada.to_csv(path_tabla4, index=False, encoding="utf-8")
    print(f"[OK] Tabla EDA guardada en: {path_tabla4}")

    return tabla_filtrada



def borrar_filas_vacias(df4:pd.DataFrame )->pd.DataFrame:
    df4_copia = df4.copy()

    datos_columnas = [c for c in df4_copia.columns if c !="username"]

    filas_vacias = df4_copia[datos_columnas].isna().all(axis=1)

    print(f"Filas vacías encontradas: {filas_vacias.sum()}")

    df4_copia=df4_copia[~filas_vacias].copy()

    path_tabla5 = OUT_DIR / "tabla_eda5.csv"
    df4_copia.to_csv(path_tabla5, index=False, encoding="utf-8")
    print(f"[OK] Tabla EDA guardada en: {path_tabla5}")

    return df4_copia

def missing_values(tabla:pd.DataFrame)-> pd.DataFrame:    # hay que arreglarlo, no funciona, no detecta los NaN
    imp = SimpleImputer(missing_values= np.nan,strategy=('mean'))
    datos_cambiados = imp.fit_transform(tabla.iloc[:,1:] ) 
    tabla.iloc[:,1:] = datos_cambiados
    


    path_tabla6= OUT_DIR / "tabla_eda6.csv"
    tabla.to_csv(path_tabla6, index=False, encoding="utf-8")
    print(f"[OK] Tabla EDA guardada en: {path_tabla6}")
    return tabla

import numpy as np
from sklearn.impute import SimpleImputer

def missing_values(tabla: pd.DataFrame) -> pd.DataFrame:
    # ddetectar los NaN y rellenar con la media de la columna
    #Ver cuántos NaN hay antes
    print("NaN antes de Simple imputer:")
    print(tabla.isna().sum())

   
    tabla = tabla.replace(["null", "None", ""], np.nan) # convierte las columnas con null, none o vacio en NaN

    columnas_numericas = tabla.select_dtypes(include=[np.number]).columns

    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    tabla[columnas_numericas] = imp.fit_transform(tabla[columnas_numericas])

  
    print("NaN después de Simple imputer:")
    print(tabla.isna().sum())

    return tabla



def correlacion(data:pd.DataFrame)->tuple[pd.DataFrame]:
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

    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= 0.9:
                if columns[j]:
                    columns[j] = False    

    selected_columns = data.columns[columns]
    print(selected_columns.shape)
    data = data[selected_columns]
    return data


def datos_procesados_eda()-> pd.DataFrame:
    archivo_inicial = medidas_validas()
    archivo_mod1 = invalidar_actividades_nulos(archivo_inicial)
    archivo_mod2 = filtrar_actividades(archivo_mod1)
    archivo_mod3 = borrar_filas_vacias(archivo_mod2)
    archivo_mod4 = missing_values(archivo_mod3)

    archivo_salida = "C:/Users/pprru/Desktop/salidas2/procesado_eda.csv"
    archivo_mod4.to_csv(archivo_salida, index=False, encoding="utf-8")
    print(f"[OK] Tabla Y guardada en: {archivo_salida}")

    return archivo_mod4

tabla_eda = datos_procesados_eda()


def construir_tabla_eda_con_diagnostico(df_eda: pd.DataFrame) -> pd.DataFrame: # tabla con la columna de diagnostico
    
    demog = pd.read_json(DEMOG_JSON)  # tabla donde esta el nombre de los usuarios con el diagnóstico
    demog_labels = demog[["user", "diagnosed"]].copy()
    demog_labels = demog_labels.rename(columns={"user": "username"})

    tabla_diag = df_eda.merge(demog_labels, on="username", how="inner")

    print(f"[OK] Tabla EDA + diagnóstico: {tabla_diag.shape}")

    # asociar yes/no/undetermined a su valor numerico
    mapa_diag = {
        "yes": 1,
        "no": 0,
        "Yes": 1,
        "No": 0,
        "undetermined": 1,  
    }

    if tabla_diag["diagnosed"].dtype == "object":
        tabla_diag["diagnosed"] = tabla_diag["diagnosed"].map(mapa_diag)
    else:
        tabla_diag["diagnosed"] = tabla_diag["diagnosed"].astype(int)

    

    return tabla_diag


def guardar_tablas(todas_filas: pd.DataFrame,salida_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    """
    Recibe la tabla de características + diagnosed (todas_filas)
    y un directorio de salida.
    Devuelve X, y y las rutas de las tres tablas creadas.
    """
    print(todas_filas)
    if todas_filas is None or todas_filas.empty:
        raise ValueError("[ERROR] 'todas_filas' está vacío. No se puede guardar nada.")


    tabla_final = todas_filas.copy()
    out_path = salida_dir / "tabla_eda_con_diagnostico.csv"
    tabla_final.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Tabla EDA + diagnóstico guardada en: {out_path}")

    # columna diagnosed (binaria)
    y = tabla_final[["diagnosed"]]
    y_out_path = salida_dir / "labels_eda_all_users.csv"
    y.to_csv(y_out_path, index=False, encoding="utf-8")
    print(f"[OK] Tabla Y guardada en: {y_out_path}")

    #  X (features) - quitamos diagnosed y columnas de identificación
    # OJO: usa los nombres correctos: ¿'user' o 'username'? ¿'epoch' existe?
    columnas_a_quitar = [c for c in ["diagnosed", "user", "username", "epoch"] if c in tabla_final.columns]

    X = tabla_final.drop(columns=columnas_a_quitar)
    X_out_path = salida_dir / "features_eda_all_users.csv"
    X.to_csv(X_out_path, index=False, encoding="utf-8")
    print(f"[OK] Tabla X guardada en: {X_out_path}")

    return X, y, X_out_path, y_out_path, out_path

#def compute_eda(base_dir: Path = BASE_DIR, demog_json: Path = DEMOG_JSON, eda_csv: Path = EDA_CSV,salida_dir: Path = OUT_DIR,) -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path]:
def compute_eda(base_dir: Path = BASE_DIR, demog_json: Path = DEMOG_JSON, eda_csv: Path = EDA_CSV,salida_dir: Path = OUT_DIR):
    salida_dir.mkdir(parents=True, exist_ok=True)

    print("\n[STEP 1] medidas_validas()")
    df2 = medidas_validas()

    print("\n[STEP 2] invalidar_actividades_nulos()")
    df3 = invalidar_actividades_nulos(df2)

    print("\n[STEP 3] filtrar_actividades()")
    df4 = filtrar_actividades(df3)

    print("\n[STEP 4] borrar_filas_vacias()")
    df5 = borrar_filas_vacias(df4)

    print("\n[STEP 5] missing_values()")
    df6 = missing_values(df5)

    print("\n[STEP 6] missing_values()")
    df6 = missing_values(df5)

    print("\n[STEP 6] construir_tabla_eda_con_diagnostico()")
    df7 = construir_tabla_eda_con_diagnostico(df6)

    print("\n[STEP 7] guardar_tablas()")
    X, y, X_out_path, y_out_path, out_path = guardar_tablas(df7, salida_dir)

    print("\n[OK] Pipeline EDA completo.")
    print(f"     - X  -> {X_out_path}")
    print(f"     - y  -> {y_out_path}")
    print(f"     - XY -> {out_path}")

    return X, y, X_out_path, y_out_path, out_path



# valores primeros y ultimos no sirven
# ver que hay suficientes valores no null

# si hay mas de dos null en una actividad ya no sirve 

# valores 0 y nulos, correlacion - seleccion de caracteristicas / outliers, curtosis 
# Main

if __name__ == "__main__":
    salida_custom = Path(r"C:\Users\pprru\Desktop\salidas_eda_nueva")
    X, y, X_path, y_path, full_path = compute_eda(salida_dir=salida_custom)
    