#%% ----------------------------------------------------------------------------------------
# Rutas básicas de los archivos del dataset

from pathlib import Path
from typing import List, Tuple

import re
import pandas as pd
import numpy as np
import mne

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns
from mne_features.univariate import compute_pow_freq_bands
from limpieza import get_eligible_users  # usa tu módulo existente

BASE_DIR   = Path(r"C:\Users\pprru\Desktop\Balladeer\balladeer_data")                 # raíz UB####
EDA_CSV    = Path(r"C:\Users\pprru\Desktop\Balladeer\balladeer_embraceplus_data.csv") # EDA/embrace
DEMOG_JSON = Path(r"C:\Users\pprru\Desktop\Balladeer\users_demographics.json")
OUT_DIR    = Path(r"C:\Users\pprru\Desktop\salidas2")                                 # salidas
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT = False  # pon True si quieres ver raw.plot(...) y raw.plot_sensors(...)

#%% -------------------------------------------------------------------------------------------------
# Lista de canales EEG Emotiv

EMOTIV_EEG_CHS = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']

#%% -------------------------------------------------------------------------------------------------------------
# Función que devuelve la línea donde empieza la cabecera con Timestamp...
# la primera línea contiene el título, versión ...
# si no la encuentra, va a devolver la primera linea no vacía
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
def _parse_sampling_rate(path: Path, default: int = 128) -> int:
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
                except Exception:
                    pass
    return eeg_hz

# Filtra los usuarios que no tengan un csv válido de Robots
# Excluye a los que tengan en el nombre eye_tracking_data o game_data
# Solo coge como válidos lo que terminen en md.pm.bp.csv y tengan epocx o epocplus en el nombre
def find_robots_eeg_csvs(user_dir: Path) -> List[Path]:
    robots = user_dir / "AttentionRobotsDesktop"
    if not robots.is_dir():
        return []
    out: List[Path] = []
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
# Carga CSV Emotiv en un RawArray de MNE
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
def preprocess_raw_fir(
    raw: mne.io.BaseRaw,
    l_freq: float = 1.0,
    h_freq: float = 60.0,  # frecuencia máxima del filtro pasa banda
    notch_hz: float | None = 50.0,
    reref: str | None = "average",
    resample_hz: float | None = 128.0,
) -> mne.io.BaseRaw:
    r = raw.copy()

    # Filtro pasa banda FIR estable + fase cero (sin desfase temporal)
    r.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_design="firwin",  # método estable
        phase="zero",
        picks="eeg",
        verbose=False,
    )

    # Notch 50 Hz (red eléctrica)
    if notch_hz:
        r.notch_filter(freqs=[notch_hz], picks="eeg", method="fir", verbose=False)

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
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=win_sec,     # tamaño ventana 4 segundos
        overlap=overlap_sec,  # tamaño del solape 2 segundos
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
def caracteristicas_bandpower_from_epochs(epochs: mne.Epochs) -> np.ndarray:
    # para sacar data se usa epochs.get_data() que devuelve un array numpy
    nd_epochs = epochs.get_data()  # shape (n_epochs, n_channels, n_times)
    sfreq = float(epochs.info["sfreq"])

    # bandas de frecuencia
    bands = np.array([0.5, 4., 8., 13., 30., 60.])  # no puede superar la mitad del sampling rate

    # parámetros del Welch

    # parámetros de Welch para mne-features:
    #claves válidas son: 'welch_n_fft', 'welch_n_per_seg', 'welch_n_overlap'
    ventana_muestras = int(sfreq * 2)   # ~2 s de ventana
    solape_muestras  = int(sfreq * 1)   # 50% solape

    psd_params = {
        "welch_n_fft": ventana_muestras,
        "welch_n_per_seg": ventana_muestras,
        "welch_n_overlap": solape_muestras,
    }

    n_epochs = nd_epochs.shape[0]
    lista_todas: List[np.ndarray] = []

    for i in range(n_epochs):
        lista = compute_pow_freq_bands(
            sfreq=sfreq,
            data=nd_epochs[i],       # (n_channels, n_times)
            freq_bands=bands,
            normalize=False,
            ratios=None,
            ratios_triu=False,
            psd_method="welch",
            log=True,
            psd_params=psd_params,   # <-- ahora SÍ usamos psd_params
        )
        lista_todas.append(lista)

    lista_todas = np.vstack(lista_todas)  # (n_epochs, n_features)
    return lista_todas

# los canales son 14 y las bandas 5, por tanto 70 características por época

#%% -------------------------------------------------------------------------------------------------------------
def construir_nombres_columnas(epochs: mne.Epochs) -> List[str]:
    channels = epochs.ch_names
    bands_labels = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    columnas: List[str] = []
    for ch in channels:
        for band in bands_labels:
            columnas.append(f"{ch}_{band}")
    return columnas

#%% -------------------------------------------------------------------------------------------------------------
def crear_tabla_caracteristicas(base_dir: Path, demog_json: Path, eda_csv: Path) -> pd.DataFrame:
    # recibe los paths de las tres carpetas
    # devuelve la tabla de las caracteristicas
    todas_filas: List[pd.DataFrame] = []  # almacena filas de todos los usuarios

    eligible_users = get_eligible_users(base_dir, demog_json, eda_csv)
    print(f"[INFO] Usuarios elegibles: {len(eligible_users)} -> {eligible_users}")

    demog = pd.read_json(demog_json)  # datos demográficos

    for user in eligible_users:
        udir = base_dir / user  # carpeta del usuario
        eeg_csvs = find_robots_eeg_csvs(udir)  # CSV EEG Robots

        if not eeg_csvs:
            print(f"[WARN] {user}: sin CSV EEG válido en Robots")
            continue

        # Buscar diagnóstico en demog
        fila_user = demog.loc[demog["user"] == user, "diagnosed"]
        if fila_user.empty:
            print(f"[WARN] {user}: no encontrado en DEMOG, se omite")
            continue

        diagnosed = str(fila_user.values[0]).lower()
        if diagnosed in ("yes", "undetermined"):
            diagnosed_bin = 1
        else:
            diagnosed_bin = 0

        for csv in eeg_csvs:
            try:
                raw = load_emotiv_raw(csv)
            except Exception as e:
                print(f"[ERROR] {user}: fallo cargando {csv.name}: {e}")
                continue

            raw_p = preprocess_raw_fir(
                raw,
                l_freq=1.0,
                h_freq=60.0,
                notch_hz=50.0,
                reref="average",
                resample_hz=128.0,
            )
            epochs = make_epochs(raw_p, win_sec=4.0, overlap_sec=2.0)

            features = caracteristicas_bandpower_from_epochs(epochs)
            n_epochs = features.shape[0]

            feat_cols = construir_nombres_columnas(epochs)
            df_feats = pd.DataFrame(features, columns=feat_cols)

            df_feats.insert(0, "epoch", np.arange(n_epochs))
            df_feats.insert(0, "user", user)
            df_feats.insert(0, "diagnosed", diagnosed_bin)

            todas_filas.append(df_feats)

            print(f"[INFO] {user} - {csv.name}: añadidas {n_epochs} ventanas con shape {features.shape}")

    tabla = pd.concat(todas_filas, ignore_index=True)
    return tabla



#%%
def datos_procesados_pow() -> pd.DataFrame:
    tabla_con_diagnostico = crear_tabla_caracteristicas(
        base_dir=BASE_DIR,
        demog_json=DEMOG_JSON,
        eda_csv=EDA_CSV,
    )

    # quitar la columna diagnosed
    tabla_sin_diag = tabla_con_diagnostico.drop(columns=["diagnosed"])

    archivo_salida = "C:/Users/pprru/Desktop/salidas2/procesado_pow.csv"
    tabla_sin_diag.to_csv(archivo_salida, index=False, encoding="utf-8")
    print(f"[OK] Tabla X (sin diagnosed) guardada en: {archivo_salida}")

    return tabla_sin_diag




def guardar_tablas(tabla_final: pd.DataFrame,salida_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path]:

    # Comprobar que no viene vacía
    if tabla_final is None or tabla_final.empty:
        raise ValueError("[ERROR] 'tabla_final' está vacía. No se puede guardar nada.")

    # Tabla entera
    out_path = salida_dir / "bandpower_robots_all_users.csv"
    tabla_final.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Tabla global guardada en: {out_path}")

    # Y (labels)
    if "diagnosed" not in tabla_final.columns:
        raise ValueError("[ERROR] La columna 'diagnosed' no está en tabla_final.")

    y = tabla_final[["diagnosed"]]
    y_out_path = salida_dir / "labels_robots_all_users.csv"
    y.to_csv(y_out_path, index=False, encoding="utf-8")
    print(f"[OK] Tabla Y guardada en: {y_out_path}")

    # X (features)
    columnas_a_quitar = [c for c in ["diagnosed", "user", "epoch"] if c in tabla_final.columns]
    X = tabla_final.drop(columns=columnas_a_quitar)
    X_out_path = salida_dir / "features_robots_all_users.csv"
    X.to_csv(X_out_path, index=False, encoding="utf-8")
    print(f"[OK] Tabla X guardada en: {X_out_path}")

    return X, y, X_out_path, y_out_path, out_path








#%% -------------------------------------------------------------------------------------------------------------
def compute_pow(base_dir: Path = BASE_DIR,demog_json: Path = DEMOG_JSON,eda_csv: Path = EDA_CSV,salida_dir: Path = OUT_DIR,) -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    # recibe las rutas de los archivos y una ruta de salida
    # devuelve X, y y la ruta de donde se encuentran las tres tablas creadas
    tabla_caracteristicas = crear_tabla_caracteristicas(base_dir, demog_json, eda_csv)
    X, y, X_out_path, y_out_path, out_path = guardar_tablas(tabla_caracteristicas, salida_dir)
    return X, y, X_out_path, y_out_path, out_path



if __name__ == "__main__":
    salida_custom = Path(r"C:\Users\pprru\Desktop\salidas_eda_nueva")
    X, y, X_path, y_path, full_path = compute_pow(salida_dir=salida_custom)
    #datos_procesados_pow()
    #tabla= crear_tabla_caracteristicas(BASE_DIR, DEMOG_JSON, EDA_CSV)
   
    


