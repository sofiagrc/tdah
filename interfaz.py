import streamlit as st
import altair as alt
import pandas as pd     # para el grafico multibarras
import math 
from pathlib import Path
import plotly.express as px
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns 


from clasificar import classifier
from clasificar import validation   
from clasificar import clasificar
from clasificar import correlacion
from clasificar import DATA_PATH
from clasificar import read_archivo

bases_datos={
    "EEG":"pow",
    "EDA":"eda",
    "COMBINADO":"comb"
}

st.title("CLASIFICACIÓN TDAH")

selected_tipo = st.selectbox("Base De Datos: ", list(bases_datos.keys()), placeholder="Selecciona una base de datos")
selected_model = st.selectbox('Modelo de Clasificación:', list(classifier.keys()),  placeholder ="Selecciona un modelo")
selected_val = st.selectbox('Modelo de validación:', list(validation.keys()), placeholder ="Selecciona un modelo")

metric_cols = ["Accuracy", "Precision", "Recall", "Specificity", "Fscore", "AUROC"] # para poder hacer la grafica sin fn, fp...

if st.button("Cargar Datos", use_container_width=True):
    df, media = clasificar(bases_datos[selected_tipo], selected_model, selected_val)

    st.subheader("Resultados por fold")    # tabla con 5 folds
    st.dataframe(df)
    

    st.subheader("Media de métricas")   # tabla con la media de los 5 folds
    st.dataframe(media)

    st.subheader("Gráfica por fold")        # grafica de los folds


   # Nos quedamos con Num_Fold + métricas
    df_metrics = df[["Num_Fold"] + metric_cols].copy()

    # Pasar a formato "largo": una fila = (fold, métrica, valor)
    df_long = df_metrics.melt(
        id_vars="Num_Fold",
        value_vars=metric_cols,
        var_name="Métrica",
        value_name="Valor"
    )

    chart = (alt.Chart(df_long).mark_bar().encode(
            x=alt.X("Num_Fold:O", title="Fold"),
            xOffset="Métrica:N",       
            y=alt.Y("Valor:Q", title="Valor métrica"),
            color=alt.Color("Métrica:N", legend=alt.Legend(title="Métrica")),
            tooltip=["Num_Fold", "Métrica", "Valor"]
        )
    )

    st.altair_chart(chart, use_container_width=True)
        

    st.subheader("Gráfica de media")
    st.bar_chart(media[metric_cols].T)     # se transpone para que en el eje x esten las metricas

    
    tabla_rangos = (                       # tabla con la media, max, min, desviacion
        df[metric_cols]
          .agg(["mean", "std", "min", "max"])
          .T
          .reset_index()
    )
    tabla_rangos.columns = ["Métrica", "Media", "Std", "Mínimo", "Máximo"]
    tabla_rangos = tabla_rangos.round(3)

    st.subheader("Resumen de métricas")
    st.dataframe(tabla_rangos)



    metricas = media[metric_cols].iloc[0]

    # listas de valor y variable
    categorias = list(metricas.index)
    valores = list(metricas.values)

    # Cerramos el polígono repitiendo el primero al final
    categorias.append(categorias[0])
    valores.append(valores[0])

    radar_df = pd.DataFrame({
        "Métrica": categorias,
        "Valor": valores
    })

    fig = px.line_polar(radar_df, r="Valor", theta="Métrica", line_close=True, color_discrete_sequence= ['red'])
    fig.update_traces(fill="toself")

    st.subheader("Radar de métricas del modelo")
    st.plotly_chart(fig, use_container_width=True)

    # matriz de confusion
    tp = df["tp"].sum()    
    fp = df["fp"].sum()
    tn = df["tn"].sum()
    fn = df["fn"].sum()

    # Matriz en orden:
    #       Pred 0   Pred 1
    # Real 0   TN      FP
    # Real 1   FN      TP
    cm = np.array([[tn, fp],
                   [fn, tp]])

    etiquetas_filas = ["Real: Negativo", "Real: Positivo"]
    etiquetas_cols  = ["Pred: Negativo", "Pred: Positivo"]

    cm_df = pd.DataFrame(cm, index=etiquetas_filas, columns=etiquetas_cols)

    st.subheader("Matriz de confusión")
    st.dataframe(cm_df)

    if(selected_tipo=="EEG"):
        ruta = DATA_PATH+"/bandpower_robots_all_users.csv"
    elif(selected_tipo=="EDA"):
        ruta = DATA_PATH+"/tabla_eda_con_diagnostico.csv"
    else:
        ruta = DATA_PATH+"/combinada.csv"

    archivo  = read_archivo(ruta)
    if ("username" in df):
        df = df.rename(columns={"username": "user"}) 
    corr = correlacion(archivo)
    sns.heatmap(corr)
    st.pyplot(plt)


