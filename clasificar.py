
#%% -------------------------------------------------------------------------------------------------------------
#  Machine Learning con scikit-learn 
# SVM
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score   
from pathlib import Path    

# LEYENDO ARCHIVOS --------------------------------------------------------------
# modifique las variables en funcion de su config

DATA_PATH = "C:/Users/pprru/Desktop/Bueno/datos"

path_demog = "C:/Users/pprru/Desktop/Balladeer/users_demographics.json"

OUTPUT_PATH = "C:/Users/pprru/Desktop/Bueno/salidas"  


def read_archivo(path: str):

    print(f"Leyendo archivo: {path}")
    print(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _get_data(tipo:str):
    if (tipo=="pow"):
        path_x = DATA_PATH+"/features_robots_all_users.csv"
        path_y = DATA_PATH+"/labels_robots_all_users.csv"
        groups = obtener_usuarios(DATA_PATH+"/bandpower_robots_all_users.csv")

    elif (tipo=="eda"):
        path_x = DATA_PATH+"/features_eda_all_users.csv"
        path_y = DATA_PATH+"/labels_eda_all_users.csv"
        groups = obtener_usuarios(DATA_PATH+"/tabla_eda_con_diagnostico.csv")
        print("C:/Users/pprru/Desktop/salidas_eda_nueva/tabla_eda_con_diagnostico.csv")

    elif (tipo=="comb"):
        path_x = DATA_PATH+"/combinada_x.csv"
        path_y = DATA_PATH+"/combinada_y.csv"
        groups = obtener_usuarios(DATA_PATH+"/combinada.csv")

    X = read_archivo(path_x)
    y = read_archivo(path_y)

    
    return X,y,groups



def obtener_usuarios(path: str):
    print(f"Leyendo archivo: {path}")
    print(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if ("username" in df):
        df = df.rename(columns={"username": "user"})   # en cada base de datos el usuario se llama de una forma, se pone user para evitar errores


    users = df.pop("user")
    return users


def normalizar_datos(X_train, X_test):
    #print("NORMALIZANDO CON STANDARDSCALER")
    scaler = StandardScaler() # zscore normalization: media 0, desviacion tipica 1 (z=(x-mean)/std)
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test= scaler.transform(X_test) # nunca se usan datos de test para ajustar el scaler
    return X_train, X_test



# PREDICCION CON Clasificadores -------------------------------------------------

classifier = {
    "SVC": svm.SVC(),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3),

}

validation = {
    "kfold":KFold(n_splits=5),
    "groupkfold":GroupKFold(n_splits=5),
    "stratifiedkfold":StratifiedKFold(n_splits=5),
    "stratifiedgroupkfold":StratifiedGroupKFold(n_splits= 5),
    }


def clasificador_unico(tipo:str, name_clf:str):

    if (name_clf in classifier.keys()):
        clf = classifier[name_clf]
    
    else:
        print("Clasificador no válido")

    X,y,groups= _get_data(tipo)

    dicc_metricas_clasificador={}
    dicc_metricas_fold={}
    encabezado = ["Clasificador","Tipo_Fold","Num_Fold","Accuracy","Precision","Recall","Specificity","Fscore","AUROC","tn","fp","fn","tp"] 
    df = pd.DataFrame(columns=encabezado)
    datos=[]

    print("\n")
    print("=============================================================")
    print(f"Aplicando clasificador: {name_clf}")
    print("=============================================================")
    print("\n") 

    for name_val, val in validation.items():  # recorre TIPOS DE FOLDS
        print("*************************************************************")
        print(f" Usando validacion: {name_val}")
        conjunto_X_train, conjunto_X_test, conjunto_y_train, conjunto_y_test=aplicar_folds(X,y,clf,val,name_val,groups)  # solo es un fold
        
        lista =[]
        num_fold=0
        for i in range (len(conjunto_X_train)):  # recorre FOLDS
            print(f"  Fold: {num_fold}")

            X_train=conjunto_X_train[i]
            X_test=conjunto_X_test[i]
            y_train=conjunto_y_train[i]
            y_test=conjunto_y_test[i]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            print(f"Calculando metricas...")
            num_fold+=1
            dicc_metricas,tn,fp,fn,tp = calcular_metricas(X_test, y_test, y_pred,name_clf,clf,num_fold)  # guarda metricas por tipo de fold 
            #print(dicc_metricas)    # esto tiene las metricas de un fold
            print(tn,fp,fn,tp)
            # ================================================================================
            # PARA CADA DATO SE CREA UNA FILA EN LA TABLA FINAL

            fila = {
                "Clasificador": name_clf,
                "Tipo_Fold": name_val,
                "Num_Fold": num_fold,
                "Accuracy": dicc_metricas[num_fold]["accuracy"],
                "Precision": dicc_metricas[num_fold]["precision"],
                "Recall": dicc_metricas[num_fold]["recall"],
                "Specificity": dicc_metricas[num_fold]["specificity"],
                "Fscore": dicc_metricas[num_fold]["fscore"],
                "AUROC": dicc_metricas[num_fold]["auroc"],
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp
            }

            datos.append(fila)

            # ================================================================================ 



            lista.append(dicc_metricas) # guarda las metricas de todos los folds
        dicc_metricas_fold[name_clf+"_"+name_val]=lista


    print("\n")

    #print(dicc_metricas_fold)  # en este punto la lista tiene las metricas de todos los folds de un tipo de FOLD
    print("\n")
    
            
    print("*************************************************************")

    df = pd.DataFrame(datos, columns=encabezado)

    nombre_archivo = "/tabla_metricas_"+name_clf+"_"+tipo+".csv"
    out_path = Path(OUTPUT_PATH+nombre_archivo)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Tabla métricas guardada en: {out_path}")



    # en datos esta ya  la tabla completa
    # hay que agrupar por clasificador y tipo de fold y sacar la media de cada uuna de las metricas
    media = df.groupby(["Clasificador","Tipo_Fold"])[["Accuracy","Precision","Recall","Specificity","Fscore","AUROC","tn","fp","fn","tp"]].mean().reset_index()   # reset_index es para que vuelva a poner lo de clasificador y tipo_fold como columnas
    #print(media)
    nombre_archivo_media = "/tabla_metricas_media_"+name_clf+"_"+tipo+".csv"
    out_path2 = Path(OUTPUT_PATH+nombre_archivo_media)
    media.to_csv(out_path2, index=False, encoding="utf-8")
    print(f"[OK] Tabla métricas guardada en: {out_path2}")

    
    return datos











def clasificar_todos(tipo:str):


    X,y,groups= _get_data(tipo)

    dicc_metricas_clasificador={}
    dicc_metricas_fold={}

    encabezado = ["Clasificador","Tipo_Fold","Num_Fold","Accuracy","Precision","Recall","Specificity","Fscore","AUROC","tn","fp","fn","tp"] 
    df = pd.DataFrame(columns=encabezado)
    datos_total=[]

    for name_clf, clf in classifier.items():      # recorre TIPOS DE CLASIFICADORES
        datos = clasificador_unico(tipo,name_clf)    # devuelve todos los tipos de folds para un clasificador

        for el in datos:
            datos_total.append(el)
        
    
    df = pd.DataFrame(datos_total, columns=encabezado)

    nombre_archivo = "/tabla_metricas_"+tipo+".csv"
    out_path = Path(OUTPUT_PATH+nombre_archivo)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Tabla métricas guardada en: {out_path}")



    # en datos_total esta ya  la tabla completa
    # hay que agrupar por clasificador y tipo de fold y sacar la media de cada uuna de las metricas
    media = df.groupby(["Clasificador","Tipo_Fold"])[["Accuracy","Precision","Recall","Specificity","Fscore","AUROC","tn","fp","fn","tp"]].mean().reset_index()   # reset_index es para que vuelva a poner lo de clasificador y tipo_fold como columnas
    #print(media)
    nombre_archivo_media = "/tabla_metricas_media_"+tipo+".csv"
    out_path2 = Path(OUTPUT_PATH+nombre_archivo_media)
    media.to_csv(out_path2, index=False, encoding="utf-8")
    print(f"[OK] Tabla métricas guardada en: {out_path2}")




    #===============================================================            


def aplicar_folds (X: pd.DataFrame , y:pd.DataFrame , clf, val, name_val,groups):

    conjunto_X_train={}
    conjunto_X_test={}
    conjunto_y_train={}
    conjunto_y_test={}
    num_fold=0

    if(name_val=="kfold"):
        for train_ix, test_ix in val.split(X):

            #print("%s %s"% (train_ix,test_ix))
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
            
            y_train = y.iloc[train_ix].values.ravel()
            y_test  = y.iloc[test_ix].values.ravel()
            X_train, X_test = normalizar_datos(X_train, X_test)

            conjunto_X_train[num_fold] = X_train
            conjunto_X_test[num_fold] = X_test
            conjunto_y_train[num_fold] = y_train
            conjunto_y_test[num_fold] = y_test
            num_fold+=1

            

    elif(name_val=="stratifiedkfold"): 
        for fold_ix, (train_ix, test_ix) in enumerate(val.split(X,y)):
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
            X_train, X_test = normalizar_datos(X_train, X_test)

            
            y_train = y.iloc[train_ix].values.ravel()
            y_test  = y.iloc[test_ix].values.ravel()

            conjunto_X_train[num_fold] = X_train
            conjunto_X_test[num_fold] = X_test
            conjunto_y_train[num_fold] = y_train
            conjunto_y_test[num_fold] = y_test
            num_fold+=1

            

            
    elif(name_val=="groupkfold" ):
        for train_ix, test_ix in val.split(X, y, groups=groups):
            #print("%s %s"% (train_ix,test_ix))
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
            X_train, X_test = normalizar_datos(X_train, X_test)


            y_train = y.iloc[train_ix].values.ravel()
            y_test  = y.iloc[test_ix].values.ravel()

            conjunto_X_train[num_fold] = X_train
            conjunto_X_test[num_fold] = X_test
            conjunto_y_train[num_fold] = y_train
            conjunto_y_test[num_fold] = y_test
            num_fold+=1

            
        


    elif name_val == "stratifiedgroupkfold":
        for train_ix, test_ix in val.split(X, y.values.ravel(), groups=groups):
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
            X_train, X_test = normalizar_datos(X_train, X_test)

            y_train = y.iloc[train_ix].values.ravel()
            y_test  = y.iloc[test_ix].values.ravel()

            conjunto_X_train[num_fold] = X_train
            conjunto_X_test[num_fold] = X_test
            conjunto_y_train[num_fold] = y_train
            conjunto_y_test[num_fold] = y_test
            num_fold+=1



    
    
    return conjunto_X_train, conjunto_X_test, conjunto_y_train, conjunto_y_test
   
#  CALCULO DE METRICAS -------------------------------------------------


def calcular_metricas(X_test, y_test, y_pred, name_clf, clf, fold):
    dicc_metricas={}

    accuracy = accuracy_score(y_test, y_pred)
    p = precision_score(y_test, y_pred, average='binary')
    r = recall_score(y_test, y_pred, average='binary')

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel().tolist()

    # Evitar división por cero en specificity SSINO DA ERROR PARA COMBINADA
    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = np.nan  # o 0.0 si prefieres

    precision, recall, fscore, support = precision_recall_fscore_support(
        y_test, y_pred, average='binary'
    )

    # CONTROLA QUE NO DE ERROR PARA KNN
    auroc = np.nan
    clases_test = np.unique(y_test)
    if len(clases_test) > 1:
        try:
            if hasattr(clf, "decision_function"):
                scores = clf.decision_function(X_test)
            elif hasattr(clf, "predict_proba"):
                scores = clf.predict_proba(X_test)[:, 1]
            else:
                scores = None

            if scores is not None:
                auroc = roc_auc_score(y_test, scores)
        except ValueError:
            auroc = np.nan

    dicc_metricas[fold] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fscore": fscore,
        "auroc": auroc,
    }

    return dicc_metricas, tn, fp, fn, tp



def estudio_demografico():

    datos_demog = pd.read_json(path_demog)
    etiquetas = datos_demog["diagnosed"].values.tolist()
    #print(etiquetas)

    etiquetas_si=0
    etiquetas_no=0
    for i in range(len(etiquetas)):
        if(etiquetas[i]=="yes" or etiquetas[i]=="undetermined"):
            etiquetas_si+=1
        else:
            etiquetas_no+=1


    print(f"Número total de usuarios: {len(etiquetas)}")
    print(f"Número de usuarios etiquetados como NO: {etiquetas_no}")
    print(f"Número de usuarios etiquetados como SI: {etiquetas_si}")

    print(f"Porcentaje NO: {etiquetas_no/len(etiquetas)}")  
    print(f"Porcentaje SI: {etiquetas_si/len(etiquetas)}")
    




# añadir funcion get_Data  --
# cambiar Paths  --
# cambiar nombre archivos salida metricas --
# añadir funcion por tipo y clasificador --
# añadir otro clasificador

if __name__ == "__main__":
   #estudio_demografico()

    #clasificador_unico("comb","KNN")
    #clasificador_unico("eda","AdaBoost")
    #clasificador_unico("pow","SVC")


    #clasificar_todos("comb")
    clasificar_todos("eda")
    #clasificar_todos("pow")



