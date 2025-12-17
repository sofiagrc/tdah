
#%% -------------------------------------------------------------------------------------------------------------
#  Machine Learning con scikit-learn 
# SVM
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns 
import os


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
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from preprocesing import read_archivo
from preprocesing import obtener_usuarios
from pathlib import Path  
from preprocesing import correlacion  
from preprocesing import eliminar_correlacion
from crear_csv_pow import guardar_tablas as guardar_tablas_pow
from crear_csv_eda import guardar_tablas as guardar_tablas_eda
from crear_combinado import limpiar_tabla

# LEYENDO ARCHIVOS --------------------------------------------------------------
# modifique las variables en funcion de su config

DATA_PATH = "C:/Users/pprru/Desktop/Bueno/datos"

path_demog = "C:/Users/pprru/Desktop/Balladeer/users_demographics.json"

OUTPUT_PATH = "C:/Users/pprru/Desktop/Bueno/salidas"  






def _get_data(tipo:str, correlacion_lim=1):

    if correlacion_lim!=1:
        texto_correlacion="_"+str(correlacion_lim)
    else:
        texto_correlacion=""

    if (tipo=="pow"):
        path_x = DATA_PATH+"/features_robots_all_users"+texto_correlacion+".csv"
        path_y = DATA_PATH+"/labels_robots_all_users"+texto_correlacion+".csv"
    elif (tipo=="eda"):
        path_x = DATA_PATH+"/features_eda_all_users"+texto_correlacion+".csv"
        path_y = DATA_PATH+"/labels_eda_all_users"+texto_correlacion+".csv"
    elif (tipo=="comb"):
        path_x = DATA_PATH+"/combinada_x"+texto_correlacion+".csv"
        path_y = DATA_PATH+"/combinada_y"+texto_correlacion+".csv"

    if not (os.path.exists(path_x) and os.path.exists(path_y)):
        # primero hay que crear los archivos
        if tipo=="pow":
            archivo_base= DATA_PATH+ "/bandpower_robots_all_users.csv"
            
        elif (tipo=="eda"):
            archivo_base = (DATA_PATH+"/tabla_eda_con_diagnostico.csv")
            
        else:
            archivo_base = DATA_PATH+ "/combinada.csv"

        print(archivo_base)
        archivo = read_archivo(archivo_base)

        print(" se ha leido el archivo")

        #print(type(archivo))
        #print(archivo)


        data, corr =correlacion(archivo)    # le puedo pasar un str o un dataframe
        # devuelve la correlacion numerica (corr) y 

        data2 = eliminar_correlacion(data,corr,correlacion_lim)

        print("hace la correlacion: ")
        #print(data2)


        #ahora hay que generar los archivos x e y segun el tipo

        if tipo=="pow" :
            X, y, path_x, path_y, out_path = guardar_tablas_pow(data2,  Path(DATA_PATH),correlacion_lim)
        
        elif tipo=="eda":
            X, y, path_x, path_y, out_path = guardar_tablas_eda(data2,  Path(DATA_PATH),correlacion_lim)
            
        else:
            X, y, path_x, path_y, out_path =limpiar_tabla(data2,correlacion_lim)
    

    # saca los grupos de las tablas estuvieran creadas o no

    if tipo=="pow":
        groups = obtener_usuarios(DATA_PATH+ "/bandpower_robots_all_users.csv")

    elif (tipo=="eda"):
        groups = obtener_usuarios(DATA_PATH+ "/tabla_eda_con_diagnostico.csv")
    
    else:
        groups = obtener_usuarios(DATA_PATH+ "/combinada.csv")

    
    X = read_archivo(path_x)
    y = read_archivo(path_y)

    return X,y,groups




def normalizar_datos(X_train, X_test):
    #print("NORMALIZANDO CON STANDARDSCALER")
    scaler = StandardScaler() # zscore normalization: media 0, desviacion tipica 1 (z=(x-mean)/std)
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test= scaler.transform(X_test) # nunca se usan datos de test para ajustar el scaler
    return X_train, X_test



# PREDICCION CON Clasificadores -------------------------------------------------

#//////////////////////------------------------------------------------------------------------------------------------------

def crear_torch(n_features, n_classes, n_hidden=64):

    return nn.Sequential(
        nn.Linear(n_features, n_hidden),
        nn.Sigmoid(),
        nn.Linear(n_hidden, n_classes)
    )



classifier = {
    "SVC": svm.SVC(),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "MLP_1": MLPClassifier(hidden_layer_sizes=(128,64,32), random_state=42),
    "MLP_2": MLPClassifier(hidden_layer_sizes=(128,256,128,64), random_state=42),
    "MLP_3": MLPClassifier(hidden_layer_sizes=(128,256,128,64,32), random_state=42),
    "Pytorch": crear_torch

    

}

validation = {
    "kfold":KFold(n_splits=5),
    "groupkfold":GroupKFold(n_splits=5),
    "stratifiedkfold":StratifiedKFold(n_splits=5),
    "stratifiedgroupkfold":StratifiedGroupKFold(n_splits= 5),
    }



def clasificador_unico(tipo:str, name_clf:str, tipo_fold:str="",correlacion_lim=1):

    if (name_clf in classifier.keys()):
        clf = classifier[name_clf]
    
    else:
        print("Clasificador no válido")

    X,y,groups= _get_data(tipo, correlacion_lim)


    
    encabezado = ["Clasificador","Tipo_Fold","Num_Fold","Accuracy","Precision","Recall","Specificity","Fscore","AUROC","tn","fp","fn","tp"] 
    df = pd.DataFrame(columns=encabezado)
    datos=[]    
    print("\n")
    print("=============================================================")
    print(f"Aplicando clasificador: {name_clf}")
    print("=============================================================")
    print("\n") 
    if (tipo_fold==""):
        for name_val, val in validation.items():  # recorre TIPOS DE FOLDS
                
                datos_por_tipo_fold = clasificador_unico_folds(X,y,clf,name_clf, val, name_val,groups)

                for el in datos_por_tipo_fold:
                    datos.append(el)
    else:
        if tipo_fold in validation.keys():
            name_val = tipo_fold
            val = validation[name_val]
            datos_por_tipo_fold = clasificador_unico_folds(X,y,clf,name_clf, val, name_val,groups)
            for el in datos_por_tipo_fold:
                datos.append(el)
        
        else:
            print("No es un tipo correcto de Fold")
            

    #print(datos)
    print("\n")
    print("*************************************************************")

    
    return datos


def clasificador_unico_folds(X,y,clf,name_clf,val,name_val,groups):

    print("*************************************************************")
    print(f" Usando validacion: {name_val}")
    conjunto_X_train, conjunto_X_test, conjunto_y_train, conjunto_y_test=aplicar_folds(X,y,clf,val,name_val,groups)  # solo es un fold
    
    lista =[]
    num_fold=0
    datos=[]
    for i in range (len(conjunto_X_train)):  # recorre FOLDS
        print(f"  Fold: {num_fold}")

        X_train=conjunto_X_train[i]
        X_test=conjunto_X_test[i]
        y_train=conjunto_y_train[i]
        y_test=conjunto_y_test[i]

        #//////////////////////////-------------------------------------------------------------------------------------
        if name_clf == "Pytorch":
        
            Xtr = torch.tensor(X_train, dtype=torch.float32)
            ytr = torch.tensor(y_train, dtype=torch.long)
            Xte = torch.tensor(X_test,  dtype=torch.float32)

            n_features = Xtr.shape[1]   # coge las colummnas de caracteristicas
            n_classes  = int(torch.unique(ytr).numel())

            model = crear_torch(n_features, n_classes, n_hidden=64)

            criterion = torch.nn.CrossEntropyLoss(reduction="sum")
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)   #se usa ADAM como optimizador

                        
            model.train()
            for epoch in range(50):
                optimizer.zero_grad()
                logits = model(Xtr)          # forward
                loss = criterion(logits, ytr)
                loss.backward()              # backward
                optimizer.step()             # update
            
            model.eval()
            with torch.no_grad():
                logits_test = model(Xte)
                y_pred = logits_test.argmax(dim=1).cpu().numpy()
                probs  = torch.softmax(logits_test, dim=1)[:, 1].cpu().numpy()

            num_fold += 1
            
        #//////////////////////////-------------------------------------------------------------------------------------

        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            num_fold += 1
            

                
        if name_clf == "Pytorch":
            dicc_metricas, tn, fp, fn, tp = calcular_metricas(
                X_test, y_test, y_pred, name_clf, model, num_fold, scores=probs
            )
        else:
            dicc_metricas, tn, fp, fn, tp = calcular_metricas(
                X_test, y_test, y_pred, name_clf, clf, num_fold
            )


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

    return datos




def clasificar(tipo:str,clasificador="",tipo_fold="", correlacion_lim=1):

    X,y,groups= _get_data(tipo,correlacion_lim)
    print("Datos filtrando cprrelacion")
    print(X)


    dicc_metricas_clasificador={}
    dicc_metricas_fold={}

    encabezado = ["Clasificador","Tipo_Fold","Num_Fold","Accuracy","Precision","Recall","Specificity","Fscore","AUROC","tn","fp","fn","tp"] 
    df = pd.DataFrame(columns=encabezado)
    datos_total=[]
    if clasificador=="":
        for name_clf, clf in classifier.items():      # recorre TIPOS DE CLASIFICADORES
            datos = clasificador_unico(tipo,name_clf,tipo_fold,correlacion_lim)    # devuelve todos los tipos de folds para un clasificador
            for el in datos:
                datos_total.append(el)
    else:
        datos = clasificador_unico(tipo,clasificador,tipo_fold, correlacion_lim)    # devuelve todos los tipos de folds para un clasificador
        for el in datos:
            datos_total.append(el)
        
    print("tabla filtrada por correlaciion datps: ")
    df = pd.DataFrame(datos_total, columns=encabezado)


    print(df)
    print("datos metricas filtrados")

    nombre_archivo,nombre_archivo_media  = obtener_nombre_archivo(clasificador, tipo_fold, tipo,correlacion_lim)
    out_path = Path(OUTPUT_PATH+nombre_archivo)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Tabla métricas guardada en: {out_path}")

    media = df.groupby(["Clasificador","Tipo_Fold"])[["Accuracy","Precision","Recall","Specificity","Fscore","AUROC","tn","fp","fn","tp"]].mean().reset_index()   # reset_index es para que vuelva a poner lo de clasificador y tipo_fold como columnas
    #print(media)
    
    out_path2 = Path(OUTPUT_PATH+nombre_archivo_media)
    media.to_csv(out_path2, index=False, encoding="utf-8")
    print(f"[OK] Tabla métricas guardada en: {out_path2}")

    return df,media




    #===============================================================            

def obtener_nombre_archivo(clasificador, tipo_fold,tipo,correlacion_lim=1):
    if correlacion_lim==1:
        texto_corr=""
    else:
        texto_corr="_"+str(correlacion_lim)

    if clasificador=="" and tipo_fold=="":
        nombre_archivo = "/tabla_metricas_"+tipo+texto_corr+".csv"
        nombre_archivo_media = "/tabla_metricas_media_"+tipo+texto_corr+".csv"
    elif clasificador=="" and tipo_fold!="":
        nombre_archivo = "/tabla_metricas_"+tipo+"_"+tipo_fold+texto_corr+".csv"
        nombre_archivo_media = "/tabla_metricas_media_"+tipo+"_"+tipo_fold+texto_corr+".csv"
    elif clasificador!="" and tipo_fold=="":
        nombre_archivo = "/tabla_metricas_"+clasificador+"_"+tipo+texto_corr+".csv"
        nombre_archivo_media = "/tabla_metricas_media_"+clasificador+"_"+tipo+texto_corr+".csv"
    else:
        nombre_archivo = "/tabla_metricas_"+clasificador+"_"+tipo+"_"+tipo_fold+texto_corr+".csv"
        nombre_archivo_media = "/tabla_metricas_media_"+clasificador+"_"+tipo+"_"+tipo_fold+texto_corr+".csv"

    return nombre_archivo,nombre_archivo_media
        





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


def calcular_metricas(X_test, y_test, y_pred, name_clf, clf, fold, scores=None):
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
            # SOLO calcular scores internos si NO vienen dados
            if scores is None:
                if hasattr(clf, "decision_function"):
                    scores = clf.decision_function(X_test)
                elif hasattr(clf, "predict_proba"):
                    scores = clf.predict_proba(X_test)[:, 1]

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
    #clasificador_unico("eda","KNN")
    #clasificador_unico("eda","KNN","kfold")
    #clasificar_todos("eda","groupkfold")


    #clasificar_todos("comb")
    #clasificar_todos("eda")
    #clasificar_todos("pow")


    #clasificar("eda","KNN")
    #clasificar("eda","SVC","groupkfold")

    #clasificar("eda")
    #clasificar("pow")
    clasificar("comb")
    clasificar("pow")
    clasificar("eda")


# %%
