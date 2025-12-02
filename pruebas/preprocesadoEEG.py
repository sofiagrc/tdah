#%% ----------------------------------------------------------------------------------------
# Rutas básicas de los archivos del dataset

from pathlib import Path
BASE_DIR   = Path(r"C:\Users\pprru\Desktop\Balladeer\balladeer_data")                 # raíz UB####
EDA_CSV    = Path(r"C:\Users\pprru\Desktop\Balladeer\balladeer_embraceplus_data.csv") # EDA/embrace
DEMOG_JSON = Path(r"C:\Users\pprru\Desktop\Balladeer\users_demographics.json")
OUT_DIR    = Path(r"C:\Users\pprru\Desktop\salidas2")                                 # salidas
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT = False  # pon True si quieres ver raw.plot(...) y raw.plot_sensors(...)

#%% -------------------------------------------------------------------------------------------------
# Importes y lista de canales EEG 
import re, pandas as pd, numpy as np, mne

EMOTIV_EEG_CHS = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']

#%% -------------------------------------------------------------------------------------------------------------
# Función que devuelve la línea donde empieza la cabecera con Timestamp... 
# la primera línea contiene el título, versión ... 
# si no la encuentra, va a devpolver la primera linea no vacia
def _detect_header_line_for_eeg_csv(path: Path) -> int:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.strip().startswith("Timestamp") and "EEG." in line:
                return i
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.strip():
                return i
    return 0

# Busca sampling rate, si no lo encuentra lo toma como default 128 Hz
# sampling rate es el número de muestras por segundo
def _parse_sampling_rate(path: Path, default=128) -> int:
    """Intenta leer 'sampling rate: ... eeg_XXX' en las ~50 primeras líneas; si no, default."""
    eeg_hz = default
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(50):
            line = f.readline()
            if not line:
                break
            m = re.search(r"sampling\s*rate\s*:\s*.*?eeg_(\d+)", line, flags=re.I)
            if m:
                try:
                    eeg_hz = int(m.group(1))
                    break
                except:
                    pass
    return eeg_hz

# Filtra los usuarios que no tengan un csv válido de Robots
# Excluye a los que tengan en el noombre eye_tracking_data o game_data
# Solo coge como válidos lo que terminen en md.pm.bp.csv y tengan epocx o epocplus en el nombre
def find_robots_eeg_csvs(user_dir: Path) -> list[Path]:
    robots = user_dir / "AttentionRobotsDesktop"
    if not robots.is_dir():
        return []
    out = []
    for p in robots.rglob("*.csv"):
        n = p.name.upper()
        if "_EYE_TRACKING_DATA_" in n or "_GAME_DATA_" in n:
            continue
        if n.endswith("MD.PM.BP.CSV") and ("_EPOCX_" in n or "_EPOCPLUS_" in n):
            out.append(p)
    return sorted(out)
#**********************************
def _make_stamp_from_csv(csv: Path) -> str:
    """Usa la carpeta numérica (timestamp) si existe; si no, el stem sin sufijo '.md.pm.bp'."""
    stamp = csv.parent.name
    return stamp if stamp.isdigit() else csv.stem.replace(".md.pm.bp", "")
#**********************************
#%% ---------------------------------------------------------------------------------------------
# En la linea de la cabecera, busca todas las columnas que empiecen por EEG. y que estén en la lista EMOTIV_EEG_CHS
# llama a la función que busca el sampling rate 
# mne.create_info crea la info de mne con los nombres de los canales, la frecuencia de muestreo y el tipo de canal (eeg)
# mne.io.RawArray crea el objeto RawArray con los datos y la info
def load_emotiv_raw(path: Path) -> mne.io.RawArray:

    header_idx = _detect_header_line_for_eeg_csv(path)
    df = pd.read_csv(path, skiprows=header_idx)
    df.columns = [c.strip() for c in df.columns]

    eeg_map = {c.split(".", 1)[1]: c for c in df.columns if c.startswith("EEG.")}
    keep = [eeg_map[ch] for ch in EMOTIV_EEG_CHS if ch in eeg_map]
    if not keep:
        raise RuntimeError(f"No se hallaron canales EEG estándar en {path.name}")

    data = df[keep].dropna(how="all").to_numpy(dtype=float).T  # (n_ch, n_samples)
    sfreq = _parse_sampling_rate(path, default=128)

    ch_names = [c.split(".", 1)[1] for c in keep]
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    # Montaje 10–20 y visualización de sensores 
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="warn")
        if PLOT:
            raw.plot_sensors(kind="topomap", show_names=False, title="Sensores (10–20)")
    except Exception:
        pass

    if PLOT:
        try:
            raw.plot(block=False, scalings="auto", title="RAW crudo")
        except Exception:
            pass

    return raw

#%% -------------------------------------------------------------------------------------------------------------
# Filtrado FIR pasa banda + notch + re-referenciado + resampling
# 
def preprocess_raw_fir(
    raw: mne.io.BaseRaw,
    l_freq: float = 1.0,
    h_freq: float = 60.0, # frecuencia máxima del filtro pasa banda
    notch_hz: float | None = 50.0,
    reref: str | None = "average",
    resample_hz: float | None = 128.0,
) -> mne.io.BaseRaw:
    r = raw.copy()

    # Filtro pasa banda FIR estable + fase cero (sin desfase temporal)
    #
    r.filter(      # aplica filtro pasa banda
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_design="firwin", # metodo estable 
        phase="zero",
        picks="eeg",
        verbose=False,
    )

    # Notch 50 Hz (red eléctrica)
    if notch_hz:
        r.notch_filter(freqs=[notch_hz], picks="eeg", method="fir", verbose=False) # si el notch es distinto de None, aplica filtro notch

    # Re-referenciado promedio
    if reref == "average":
        r.set_eeg_reference("average", projection=False, verbose=False)

    # Resampling si hace falta
    if resample_hz and abs(r.info["sfreq"] - resample_hz) > 1e-6:
        r.resample(resample_hz, npad="auto", verbose=False)

    if PLOT:
        try:
            r.plot(block=False, scalings="auto", title="RAW preprocesado (FIR+notch+ref+resample)")
        except Exception:
            pass

    return r

#%% -------------------------------------------------------------------------------------------------------------
# Segmentación en épocas de duración win_sec y solape overlap_sec
def make_epochs(raw: mne.io.BaseRaw, win_sec: float = 4.0, overlap_sec: float = 2.0) -> mne.Epochs:
    epochs = mne.make_fixed_length_epochs(  # divide la señal en fragmentos iguales 
        raw,
        duration=win_sec,  #tamaño ventana 4 segundos
        overlap=overlap_sec, # tamaño del solape de 2 segundos
        preload=True,
        verbose=False,
    )
    sf = raw.info["sfreq"]
    starts = (epochs.events[:, 0] / sf)
    ends = starts + win_sec
    epochs.metadata = pd.DataFrame({"t_start_s": starts, "t_end_s": ends})
    return epochs


#%% -------------------------------------------------------------------------------------------------------------
# Cálculo de características de bandpower usando mne-features
# calcular potencias usando compute_pow_freq_bands
from mne_features.univariate import compute_pow_freq_bands

def caracteristicas_bandpower_from_epochs(epochs) -> np.ndarray:
    # freq bands, data?
    # para sacar data se usa epochs.get_data() que devuelve un array numpy con las épocas en la función anterior
    nd_epochs= epochs.get_data()  # shape (n_epochs, n_channels, n_times) 
    
    # print(f"[INFO] Shape de los datos de las épocas: {nd_epochs.shape}")
    # print(f"[INFO] Shape de los datos de las épocas: {nd_epochs[0].shape}")
    sfreq = float(epochs.info["sfreq"]) 
    bands = np.array([0.5, 4., 8., 13., 30., 60.])   #por la frecuencia de Nyquist no puede superar la mitad del sampling rate

    # (opcional) parámetros del Welch, ajusta si quieres más/menos resolución
    psd_params = {
        "n_fft": int(sfreq * 2),       # ~2 s de ventana
        "n_overlap": int(sfreq * 1)    # 50% solape
    }

    n_epochs = nd_epochs.shape[0]

    lista_todas=[] # va a guardar las listas de cada una de las épocas

    for i in range(n_epochs):  # recorre todas las épocas

        #print(f"[INFO] Procesando época {i+1}/{n_epochs}")
        lista = compute_pow_freq_bands(
        sfreq =  sfreq,   # es el sampling rate  (se saca en funciones anteriores)
        data =nd_epochs[i],      # al poner aqui i se recorren todas las épocas
        freq_bands=bands, # bandas de frecuencia 
        normalize=False,
        ratios=None, 
        ratios_triu=False, # si pone false se incluyen todas las características
        psd_method='welch', # método para calcular la densidad espectral de potencia
        log=True, # si se quiere en escala logarítmica
        psd_params=None # parámetros adicionales para el cálculo del PSD
        )

        lista_todas.append(lista)

    #print(f"[INFO] Shape de la lista: {lista.shape} {lista}")
    lista_todas = np.vstack(lista_todas)  # apila todas las listas en una matriz numpy  

    return lista_todas  # devuelve un array numpy con las características calculadas


#los canales son 14 y las bandas 5, por tanto 70 características por época

#%%
def construir_nombres_columnas(epochs) -> list[str]:

    channels = epochs.ch_names
    bands_labels = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    columnas = []
    for ch in channels:
        for band in bands_labels:
            columnas.append(f"{ch}_{band}")
    return columnas

#%% -------------------------------------------------------------------------------------------------------------
from limpieza import get_eligible_users  # usa tu módulo existente

todas_filas = []  #almacena filas (todas las ventanas) de todos los usuarios

eligible_users = get_eligible_users(BASE_DIR, DEMOG_JSON, EDA_CSV)

print(f"[INFO] Usuarios elegibles: {len(eligible_users)} -> {eligible_users}")


demog = pd.read_json(DEMOG_JSON) # en demog se guarda un dataframe con los datos demográficos


lista_user=[]
for user in eligible_users:
    udir = BASE_DIR / user # se crea un nuevo path para cada usuario concreto
    eeg_csvs = find_robots_eeg_csvs(udir) #vuelve a comprobar que tiene el  csv con el EEG
    if not eeg_csvs:
        print(f"[WARN] {user}: sin CSV EEG válido en Robots")
        continue

    for csv in eeg_csvs: #recorre todos los csvs encontrados de un usuario
        try:
            raw = load_emotiv_raw(csv)   # se hace la carga del csv
        except Exception as e:
            print(f"[ERROR] {user}: fallo cargando {csv.name}: {e}")
            continue

        # si ha podido cargar el raw, se preprocesa y se hacen las épocas
        raw_p = preprocess_raw_fir(   # se preproces el raw (aplica los filtros)
            raw, l_freq=1.0, h_freq=60.0, notch_hz=50.0, reref="average", resample_hz=128.0
        )
        epochs = make_epochs(raw_p, win_sec=4.0, overlap_sec=2.0)

        features = caracteristicas_bandpower_from_epochs(epochs)
        print(features)

        # features: matriz (n_epochs, n_features)
        # shape  devuelve una tupla (filas, columnas) // ahora nos devuelve el numero de filas
        n_epochs = features.shape[0] # cada fila de features es un epoch // cada columna son caracteristicas (bandas*canales)

        # 1) Nombres de columnas canal_banda
        feat_cols = construir_nombres_columnas(epochs)

        # Creamos un DataFrame con estas features y sus nombres
        df_feats = pd.DataFrame(features, columns=feat_cols)


        # columnas iniciales con nombre de usuario, nombre de archivo y número de época
        df_feats.insert(0, "epoch", np.arange(n_epochs))
        df_feats.insert(0, "user", user) 

        for i in range(n_epochs):
            lista_user.append(user)
        
        

       

        diagnosed= demog.loc[demog["user"]==user, "diagnosed"].values[0] # localiza un usuario concreto y saca su diagnóstico
        # el .loc busca en la columna user el usuario concreto y devuelve su diagnóstico 

        # hay que poner 1(yes) o 0(no) undetermined (1) 
        if diagnosed.lower() in ("yes", "undetermined"):
            diagnosed = 1
        else:   
            diagnosed = 0

        df_feats.insert(0,"diagnosed", diagnosed)  # añade la columna de diagnóstico al dataframe

        # Guardamos estas filas en la lista global
        todas_filas.append(df_feats)

        print(f"[INFO] {user} - {csv.name}: añadidas {n_epochs} ventanas con shape {features.shape}")



#%% -------------------------------------------------------------------------------------------------------------
# La y va a estar compuesta de las dos primeras columnas de la tabla, user y diagnosed.
# la x va a ser el resto de columnas, las características de bandpower con el usuario
# El siguiente paso es guardar las tablas X e Y por separado
if todas_filas:
    # Tabla entera con todos los datos
    tabla_final = pd.concat(todas_filas, ignore_index=True) # concatena todas las filas en una sola tabla
    out_path = BASE_DIR / "bandpower_robots_all_users.csv"
    tabla_final.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Tabla global guardada en: {out_path}")

    # Y(user, diagnosed)
    y = tabla_final[["diagnosed"]]       
    y_out_path = BASE_DIR / "labels_robots_all_users.csv"
    y.to_csv(y_out_path, index=False, encoding="utf-8")
    print(f"[OK] Tabla Y guardada en: {y_out_path}")

    # X(user, características)
    X = tabla_final.drop(columns=["diagnosed","user","epoch"])
    X_out_path = BASE_DIR / "features_robots_all_users.csv"
    X.to_csv(X_out_path, index=False, encoding="utf-8")
    print(f"[OK] Tabla X guardada en: {X_out_path}")

#===============================================================================================================
#===============================================================================================================
#===============================================================================================================
#===============================================================================================================
#===============================================================================================================

#%% -------------------------------------------------------------------------------------------------------------
#  Machine Learning con scikit-learn 
# SVM
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split( # divide los datos en conjunto de entrenamiento y test
    X, y, test_size=0.33, random_state=42)

#X_train

#y_train

#X_test

#y_test

#train_test_split(y, shuffle=False)

# NORMALIZACIÓN STANDARSCALER
# Antes de normalizar, guardo una copia
X_train_raw = X_train.copy()

print("NORMALIZANDO CON STANDARDSCALER")
#print(f"Sin Normalizar:{X_train} ")
scaler = StandardScaler()
scaler.fit(X_train)
#print(scaler.mean_)
X_train=scaler.transform(X_train)
X_test= scaler.transform(X_test)
print(f"Normalizada:{X_train} ")


#%%
# REPRESENTACION GRÁFICA---------------------------------------------
# Cogemos la primera columna del dataset
col_name = X_train_raw.columns[0] 

col_name = X_train_raw.columns[0] 
print(f"Columna 0: {col_name}")
print(f"Antes de Normalizar --> Media: {X_train_raw[col_name].values.mean()} Std {X_train_raw[col_name].values.std()}")


# Señal original y normalizada de esa columna
senal_original = X_train_raw[col_name].values      # antes de normalizar
senal_norm = X_train[:, 0]                         # después de normalizar (X_train es ya array)
print(f"Después de Normalizar --> Media: {senal_norm.mean()} Std {senal_norm.std()}")
# Eje X = índice de muestra
muestras = range(len(senal_original))

plt.figure(figsize=(10, 4))

# Señal original
plt.subplot(1, 2, 1)
plt.plot(muestras, senal_original)
plt.title(f"Señal original - {col_name}")
plt.xlabel("Muestras")
plt.ylabel("Valor")
plt.grid(True)

# Señal normalizada
plt.subplot(1, 2, 2)
plt.plot(muestras, senal_norm)
plt.title(f"Señal normalizada - {col_name}")
plt.xlabel("Muestras")
plt.ylabel("Valor normalizado")
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------


clf = svm.SVC()
clf.fit(X_train, y_train) 


#%% ACCURACY 
from sklearn.metrics import accuracy_score 
y_pred = clf.predict(X_test)
y_true = y_test.values.ravel()  # convertir a array 1D
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")
# %% PRECISION ---------------------------------------------------------------------------------------------------
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
from sklearn.metrics import precision_score
y_true = y_test.values.ravel()  # convertir a array 1D
y_pred= clf.predict(X_test)
p1 = precision_score(y_true, y_pred, average='macro')
p2 = precision_score(y_true, y_pred, average='micro')
p3 = precision_score(y_true, y_pred, average='weighted')
p4= precision_score(y_true, y_pred, average=None)
print(f"Macro Precision: {p1}")
print(f"Micro Precision: {p2}")   
print(f"Weighted Precision: {p3}")
print(f"Precision por clase (None){p4}")

# %% RECALL -----------------------------------------------------------------------------------------------------
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
from sklearn.metrics import recall_score
y_true = y_test.values.ravel()  # convertir a array 1D
y_pred= clf.predict(X_test)
r1 = recall_score(y_true, y_pred, average='macro')
r2 = recall_score(y_true, y_pred, average='micro')
r3 = recall_score(y_true, y_pred, average='weighted')
r4= recall_score(y_true, y_pred, average=None)
print(f"Macro Recall: {r1}")
print(f"Micro Recall: {r2}")   
print(f"Weighted Recall: {r3}")
print(f"Recall por clase (None): {r4}")
# %% SPECIFICITY -------------------------------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix
y_true = y_test.values.ravel()  # convertir a array 1D
y_pred= clf.predict(X_test)
matrix= confusion_matrix(y_true, y_pred, normalize='all')
tn,fp,fn,tp = confusion_matrix(y_true, y_pred).ravel().tolist()
specificity = tn / (tn + fp)
print(f"Specificity: {specificity}")

#%% FSCORE --------------------------------------------------------------------------------------------------------
from sklearn.metrics import precision_recall_fscore_support
y_true = y_test.values.ravel()  # convertir a array 1D
y_pred= clf.predict(X_test)
precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='macro')
print(f"Macro Fscore --> Precision: {precision}, Recall: {recall}, F-score: {fscore}, Support: {support}")
precision_recall_fscore_support(y_true, y_pred, average='micro')
print(f"Micro Fscore--> Precision: {precision}, Recall: {recall}, F-score: {fscore}, Support: {support}")
precision_recall_fscore_support(y_true, y_pred, average='weighted')
print(f"Weighted Fscore --> Precision: {precision}, Recall: {recall}, F-score: {fscore}, Support: {support}")


# %% AUROC -------------------------------------------------------------------------------------------------------
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

#roc_auc_score(y_true, clf.predict_proba(X_test)[:, 1]) # tiene que ser un clasificador de decisión o que de probabilidades cvm no lo hace
auroc= roc_auc_score(y_true, clf.decision_function(X_test))
print(f"AUROC: {auroc}")


#===============================================================================================================
#===============================================================================================================
#===============================================================================================================
#===============================================================================================================
#===============================================================================================================

# %% ---------------------------------------------------K-FOLD
from sklearn.model_selection import KFold
# hhacer tabla con los valores, no direcetamente la media

tabla_final_medias=[]

fila=[]
kf = KFold(n_splits=5)

acc=[]
prec=[]
rec=[]
spec=[]
fsc=[]
aur=[]

for train_ix, test_ix in kf.split(X):

    #print("%s %s"% (train_ix,test_ix))
    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
    #print(X_train, X_test, y_train, y_test)
        
    clf = svm.SVC()
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    acc.append(accuracy)
    #print(f"Accuracy: {accuracy}")

    p1 = precision_score(y_test, y_pred, average='macro')
    prec.append(p1)
    #print(f"Macro Precision: {p1}")
   
    r1 = recall_score(y_test, y_pred, average='macro')
    rec.append(r1)
    #print(f"Macro Recall: {r1}")

    matrix= confusion_matrix(y_test, y_pred, normalize='all')
    tn,fp,fn,tp = confusion_matrix(y_test, y_pred).ravel().tolist()
    specificity = tn / (tn + fp)
    spec.append(specificity)
    #print(f"Specificity: {specificity}")

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='macro')
    fsc.append(fscore)
    #print(f"Macro Fscore --> Precision: {precision}, Recall: {recall}, F-score: {fscore}, Support: {support}")

    auroc= roc_auc_score(y_test, clf.decision_function(X_test))
    aur.append(auroc)
    #print(f"AUROC: {auroc}")

    fila.append({
        "Accuracy": accuracy,
        "Precision": p1,
        "Recall": r1,
        "Specificity": specificity,
        "FScore": fscore,
        "AUROC": auroc
    }
    )
    df_folds = pd.DataFrame(fila)


   
print("TABLA K-FOLD")
print(df_folds)
print(f"MEDIA ACCURACY: {sum(acc)/len(acc)}")
print(f"MEDIA PRECISION: {sum(prec)/len(prec)}")
print(f"MEDIA RECALL: {sum(rec)/len(rec)}")
print(f"MEDIA SPECIFICITY: {sum(spec)/len(spec)}")
print(f"MEDIA FSCORE: {sum(fsc)/len(fsc)}")
print(f"MEDIA AUROC: {sum(aur)/len(aur)}")

tabla_final_medias.append({
    "Media Accuracy":sum(acc)/len(acc),
    "Media Precision": sum(prec)/len(prec),
    "Media Recall": sum(rec)/len(rec),
    "Media Specificity": sum(spec)/len(spec),
    "Media FScore": sum(fsc)/len(fsc),
    "Media AUROC": sum(aur)/len(aur)
})
df_folds2 = pd.DataFrame(tabla_final_medias)



# %% -------------------------------------------------- GROUP K-FOLD
from sklearn.model_selection import GroupKFold
groups = lista_user
gkf = GroupKFold(n_splits=5)

fila2=[]

acc=[]
prec=[]
rec=[]
spec=[]
fsc=[]
aur=[]

for train_ix, test_ix in gkf.split(X, y, groups=groups):
    #print("%s %s"% (train_ix,test_ix))
    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
    #print(X_train, X_test, y_train, y_test)
        
    clf = svm.SVC()
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    #print(f"Accuracy: {accuracy}")
    acc.append(accuracy)

    p1 = precision_score(y_test, y_pred, average='macro')
    prec.append(p1)
    #print(f"Macro Precision: {p1}")
   
    r1 = recall_score(y_test, y_pred, average='macro')
    rec.append(r1)
    #print(f"Macro Recall: {r1}")

    matrix= confusion_matrix(y_test, y_pred, normalize='all')
    tn,fp,fn,tp = confusion_matrix(y_test, y_pred).ravel().tolist()
    specificity = tn / (tn + fp)
    spec.append(specificity)
    #print(f"Specificity: {specificity}")

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='macro')
    fsc.append(fscore)
    #print(f"Macro Fscore --> Precision: {precision}, Recall: {recall}, F-score: {fscore}, Support: {support}")

    auroc= roc_auc_score(y_test, clf.decision_function(X_test))
    aur.append(auroc)
    #print(f"AUROC: {auroc}")
    fila2.append({
        "Accuracy": accuracy,
        "Precision": p1,
        "Recall": r1,
        "Specificity": specificity,
        "FScore": fscore,
        "AUROC": auroc
    }
    )
    df_folds = pd.DataFrame(fila2)

print("TABLA GROUP K-FOLD")
print(df_folds)

print(f"MEDIA ACCURACY: {sum(acc)/len(acc)}")
print(f"MEDIA PRECISION: {sum(prec)/len(prec)}")
print(f"MEDIA RECALL: {sum(rec)/len(rec)}")
print(f"MEDIA SPECIFICITY: {sum(spec)/len(spec)}")
print(f"MEDIA FSCORE: {sum(fsc)/len(fsc)}")
print(f"MEDIA AUROC: {sum(aur)/len(aur)}")

tabla_final_medias.append({
    "Media Accuracy":sum(acc)/len(acc),
    "Media Precision": sum(prec)/len(prec),
    "Media Recall": sum(rec)/len(rec),
    "Media Specificity": sum(spec)/len(spec),
    "Media FScore": sum(fsc)/len(fsc),
    "Media AUROC": sum(aur)/len(aur)
})
df_folds2 = pd.DataFrame(tabla_final_medias)


# %% -----------------------------------------STRATIFIED KFOLD

from sklearn.model_selection import StratifiedKFold
skf= StratifiedKFold(n_splits=5)
skf.get_n_splits(X,y)

fila3=[]

for fold_ix, (train_ix, test_ix) in enumerate(skf.split(X,y)):
    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

    clf = svm.SVC()
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    #print(f"Accuracy: {accuracy}")
    acc.append(accuracy)

    p1 = precision_score(y_test, y_pred, average='macro')
    prec.append(p1)
    #print(f"Macro Precision: {p1}")
   
    r1 = recall_score(y_test, y_pred, average='macro')
    rec.append(r1)
    #print(f"Macro Recall: {r1}")

    matrix= confusion_matrix(y_test, y_pred, normalize='all')
    tn,fp,fn,tp = confusion_matrix(y_test, y_pred).ravel().tolist()
    specificity = tn / (tn + fp)
    spec.append(specificity)
    #print(f"Specificity: {specificity}")

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='macro')
    fsc.append(fscore)
    #print(f"Macro Fscore --> Precision: {precision}, Recall: {recall}, F-score: {fscore}, Support: {support}")

    auroc= roc_auc_score(y_test, clf.decision_function(X_test))
    aur.append(auroc)
    #print(f"AUROC: {auroc}")
    fila3.append({
        "Accuracy": accuracy,
        "Precision": p1,
        "Recall": r1,
        "Specificity": specificity,
        "FScore": fscore,
        "AUROC": auroc
    }
    )
    df_folds = pd.DataFrame(fila3)

print("TABLA STRATIFIED K-FOLD")
print(df_folds)

print(f"MEDIA ACCURACY: {sum(acc)/len(acc)}")
print(f"MEDIA PRECISION: {sum(prec)/len(prec)}")
print(f"MEDIA RECALL: {sum(rec)/len(rec)}")
print(f"MEDIA SPECIFICITY: {sum(spec)/len(spec)}")
print(f"MEDIA FSCORE: {sum(fsc)/len(fsc)}")
print(f"MEDIA AUROC: {sum(aur)/len(aur)}")

tabla_final_medias.append({
    "Media Accuracy":sum(acc)/len(acc),
    "Media Precision": sum(prec)/len(prec),
    "Media Recall": sum(rec)/len(rec),
    "Media Specificity": sum(spec)/len(spec),
    "Media FScore": sum(fsc)/len(fsc),
    "Media AUROC": sum(aur)/len(aur)
})
df_folds2 = pd.DataFrame(tabla_final_medias)



# %% ----------------------------------------- STRATIFIED GROUP K-FOLD
from sklearn.model_selection import StratifiedGroupKFold
groups = lista_user
sgkf = StratifiedGroupKFold(n_splits= 5)
fila4=[]


for train,test in sgkf.split(X, y, groups=groups):
    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

    clf = svm.SVC()
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    #print(f"Accuracy: {accuracy}")
    acc.append(accuracy)

    p1 = precision_score(y_test, y_pred, average='macro')
    prec.append(p1)
    #print(f"Macro Precision: {p1}")
   
    r1 = recall_score(y_test, y_pred, average='macro')
    rec.append(r1)
    #print(f"Macro Recall: {r1}")

    matrix= confusion_matrix(y_test, y_pred, normalize='all')
    tn,fp,fn,tp = confusion_matrix(y_test, y_pred).ravel().tolist()
    specificity = tn / (tn + fp)
    spec.append(specificity)
    #print(f"Specificity: {specificity}")

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='macro')
    fsc.append(fscore)
    #print(f"Macro Fscore --> Precision: {precision}, Recall: {recall}, F-score: {fscore}, Support: {support}")

    auroc= roc_auc_score(y_test, clf.decision_function(X_test))
    aur.append(auroc)
    #print(f"AUROC: {auroc}")
    fila4.append({
        "Accuracy": accuracy,
        "Precision": p1,
        "Recall": r1,
        "Specificity": specificity,
        "FScore": fscore,
        "AUROC": auroc
    }
    )
    df_folds = pd.DataFrame(fila4)

print("TABLA STRATIFIED GROUP K-FOLD")
print(df_folds)

print(f"MEDIA ACCURACY: {sum(acc)/len(acc)}")
print(f"MEDIA PRECISION: {sum(prec)/len(prec)}")
print(f"MEDIA RECALL: {sum(rec)/len(rec)}")
print(f"MEDIA SPECIFICITY: {sum(spec)/len(spec)}")
print(f"MEDIA FSCORE: {sum(fsc)/len(fsc)}")
print(f"MEDIA AUROC: {sum(aur)/len(aur)}")

tabla_final_medias.append({
    "Media Accuracy":sum(acc)/len(acc),
    "Media Precision": sum(prec)/len(prec),
    "Media Recall": sum(rec)/len(rec),
    "Media Specificity": sum(spec)/len(spec),
    "Media FScore": sum(fsc)/len(fsc),
    "Media AUROC": sum(aur)/len(aur)
})


df_folds2 = pd.DataFrame(tabla_final_medias)
renombrada = df_folds2.rename(index={
    0: "K-FOLD",
    1: "Group K-FOLD",
    2: "Stratified K-FOLD",
    3: "Stratified Group K-FOLD"
})

print(renombrada)

#%% -------------------------------------------------------------------------------------------------------------------------
print(lista_user)
