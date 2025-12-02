#%%
import re
import json
from pathlib import Path
import argparse
import pandas as pd

#%% -------------------------------------------------------------------------------------------------------------

USER_DIR_RE = re.compile(r"^UB\d{4}$", re.IGNORECASE)

#%% -------------------------------------------------------------------------------------------------------------

# Función que devuelve True si en la carpeta de unn usuario, dento de Robots hay un EEG válido.

def _eeg_ok_in_robots(user_dir: Path) -> bool:
    robots_dir = user_dir / "AttentionRobotsDesktop"
    if not robots_dir.is_dir():
        return False
    for p in robots_dir.rglob("*.csv"):
        name = p.name.upper()
        if "_EYE_TRACKING_DATA_" in name or "_GAME_DATA_" in name:
            continue
        if name.endswith("MD.PM.BP.CSV") and ("_EPOCX_" in name or "_EPOCPLUS_" in name):
            return True
    return False

#%% -------------------------------------------------------------------------------------------------------------

# Función que crea una lista con el id de los usuarios que tienene todos los campos requeridos en el JSON de demografía.

def _demog_users_with_required_fields(json_path: Path, required=("user","diagnosed","age","gender")) -> set[str]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = list(data.values())

    keep = set()
    for rec in data:
        if not all((rec.get(k) is not None) and (str(rec.get(k)).strip() != "") for k in required):
            continue
        u = str(rec.get("user")).strip()
        if u:
            keep.add(u)
    return keep

#%% -------------------------------------------------------------------------------------------------------------

# Devuelve conjunto de usuarios con valores en Robots_eda_values en el CSV de EDA.

def _users_with_robots_eda_values(csv_path: Path) -> set[str]:
    df = pd.read_csv(csv_path, sep=";", engine="python")
    if "username" not in df.columns or "Robots_eda_values" not in df.columns:
        raise RuntimeError("Faltan columnas 'username' o 'Robots_eda_values' en el CSV EDA.")
    ok = df["Robots_eda_values"].astype(str).str.strip().ne("") & df["Robots_eda_values"].notna()
    return set(df.loc[ok, "username"].astype(str))

#%% -------------------------------------------------------------------------------------------------------------

#Devuelve lista ordenada de usuarios que cumplen:
    #  1) EEG Robots válido (EPOCX/EPOCPLUS, MD.PM.BP.csv)
    #  2) Demografía con campos requeridos
    #  3) EDA Robots con valores

def get_eligible_users(base_dir: Path, demog_json: Path, eda_csv: Path) -> list[str]:
    
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"No existe la ruta base: {base_dir}")

    # 1) EEG Robots válido
    users_with_eeg = {
        d.name for d in base_dir.iterdir()
        if d.is_dir() and USER_DIR_RE.match(d.name) and _eeg_ok_in_robots(d)
    }

    # 2) Demografía OK
    users_demog = _demog_users_with_required_fields(demog_json)

    # 3) EDA Robots con valores
    users_eda = _users_with_robots_eda_values(eda_csv)

    # Intersección final
    eligible = sorted(users_with_eeg & users_demog & users_eda)
    return eligible



# %%
