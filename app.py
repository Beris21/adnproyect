import streamlit as st # type: ignore
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay   # type: ignore
from matplotlib import pyplot as plt    
import seaborn as sns # type: ignore
import tensorflow as tf
import tensorflow.keras as kr                           #type: ignore
from tensorflow.keras.layers import Input               #type: ignore
from tensorflow.keras.models import Sequential          #type: ignore
from tensorflow.keras.optimizers import SGD             #type: ignore
from tensorflow.keras.optimizers import Adam             #type: ignore
from sklearn.model_selection import train_test_split    #type: ignore   
from tensorflow.keras.utils import to_categorical       #type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay   #type: ignore
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #type: ignore
import pandas as pd
from sklearn.linear_model import LinearRegression                                   # type: ignore
from sklearn.model_selection import train_test_split                                # type: ignore
from sklearn.metrics import mean_squared_error                                      # type: ignore
from sklearn.linear_model  import LogisticRegression                                # type: ignore
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay                # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

def separar_variables(data):
    df = data
    df_numericas = df.select_dtypes(include=['int64', 'float64'])
    df_categoricas = df.select_dtypes(include=["object"])
    df_dummies = pd.get_dummies(df_categoricas, drop_first=True).astype(int)
    df_final = pd.concat([df_numericas, df_dummies], axis=1)
    y = df_final["Fraud_Label"]
    
    variables = df_final.drop('Fraud_Label', axis=1)
    
    scaler = StandardScaler()
    df_escalado = pd.DataFrame(scaler.fit_transform(variables), columns=variables.columns)
    df1 = pd.concat([df_escalado, y], axis=1)
    return df1

def linear_regresion(data):
    df = data
    X = df.drop(['Fraud_Label'], axis = 1)
    y = df["Fraud_Label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 2025)
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    matriz = confusion_matrix(y_test, y_pred)
    class_names = ['No Fraude', 'Fraude']
    disp = ConfusionMatrixDisplay(matriz, display_labels=class_names)
    disp.plot()
    
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    fsc = f1_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spe = float(tn / (tn + fp))

    #print(f"Accuracy: ", round(acc*100, 2), '%')
    #print(f"Precision: ", round(pre*100, 2), '%')
    #print(f"Recall: ", round(rec*100, 2), '%')
    #print(f"F1 Score: ", round(fsc*100, 2), '%')
    #print("Specificity: ", round(spe*100, 2), '%')

    metricas_1 = [acc, pre, rec, spe, fsc]

    nombres = ['Accuracy', 'Precision', 'Sensitivity', 'F-Score', 'Specificity']
    
        # Crear y retornar DataFrame de métricas
    df_metricas = pd.DataFrame({
        'Métrica': nombres,
        'Valor': [f'{round(m*100, 2)}%' for m in metricas_1]
    })
    return (disp, df_metricas)

def boxplot(df: pd.DataFrame) -> list:
    """
    Genera boxplots de variables numéricas (excluyendo booleanas y binarias 0/1) y devuelve una lista de figuras matplotlib.

    Parámetros:
    - df: DataFrame de pandas.

    Retorna:
    - Lista de objetos matplotlib.figure.Figure para cada variable.
    """
    figures = []
    # Seleccionar sólo columnas numéricas relevantes
    potential_cols = df.select_dtypes(include="number").columns
    cols = [col for col in potential_cols
            if df[col].dtype != 'bool'
            and not set(df[col].dropna().unique()).issubset({0, 1})]

    for col in cols:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.set_title(f'Boxplot variable {col}')
        sns.boxplot(data=df[col], orient='v', ax=ax, color='green')
        figures.append(fig)
    return figures

@st.cache_data
def quitar_outliers(df: pd.DataFrame, columns: list = None, factor: float = 1.5) -> pd.DataFrame:
    # Determinar columnas a procesar
    cols = ['Transaction_Amount', 'Account_Balance', 'Daily_Transaction_Count', 'Avg_Transaction_Amount_7d',
            'Failed_Transaction_Count_7d', 'Card_Age',
            'Transaction_Distance', 'Risk_Score']

    # Copiar DataFrame para no modificar el original
    filtered_df = df.copy()

    # Aplicar filtrado para cada columna
    for col in cols:
        q1 = filtered_df[col].quantile(0.25)
        q3 = filtered_df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        # Filtrar valores dentro de los límites
        filtered_df = filtered_df[(filtered_df[col] >= lower_bound) & (filtered_df[col] <= upper_bound)]
    return filtered_df
# Configuración de página

def red_neuronal(data):
    df = data
    X = df.drop(['Fraud_Label'], axis = 1)
    y = df["Fraud_Label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 2025)
    
    n0 = X.shape[1]
    n1 = 50
    n2 = 12
    n3 = 1
    nn = [n0, n1, n2,n3]
    
    model = kr.Sequential()
    model.add(Input(shape = (nn[0],)))
    model.add(kr.layers.Dense(nn[1], activation = 'relu'))
    model.add(kr.layers.Dense(nn[2], activation = 'relu'))
    model.add(kr.layers.Dense(nn[3], activation = 'sigmoid'))
    
    lr = 0.0005
    
    model.compile(loss='binary_crossentropy',
                optimizer = kr.optimizers.Adam(learning_rate=lr),
                metrics=['mae', 'mse']
                )
    entrenamiento = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.4)
    
    #Acceder a los valores de loss y val_loss (pérdida de entrenamiento y de validación)
    loss = entrenamiento.history['loss']
    val_loss = entrenamiento.history['val_loss']

    #Graficar la pérdida a lo largo de las épocas
    epochs = range(1, len(loss) + 1)
    
    
    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    
    matriz = confusion_matrix(y_test, y_pred)
    class_names = ['No Fraude', 'Fraude']

    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=class_names)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    fsc = f1_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spe = float(tn / (tn + fp))

    print(f"Accuracy: ", round(acc*100, 2), '%')
    print(f"Precision: ", round(pre*100, 2), '%')
    print(f"Recall: ", round(rec*100, 2), '%')
    print(f"F1 Score: ", round(fsc*100, 2), '%')
    print("Specificity: ", round(spe*100, 2), '%')

    metricas_1 = [acc, pre, rec, fsc, spe]

    nombres = ['Accuracy', 'Precision', 'Sensitivity', 'F-Score', 'Specificity']
        
    df_metricas = pd.DataFrame({
        'Métrica': nombres,
        'Valor': [f'{round(m*100, 2)}%' for m in metricas_1]
    })
    
    return(epochs, loss, val_loss, df_metricas, disp)

st.set_page_config(page_title="Aplicación Para la Detección de Fraude Bancario")

# Inicializar paso de la aplicación
if 'step' not in st.session_state:
    st.session_state.step = 1

# Paso 1: Carga, previsualización y limpieza
if st.session_state.step == 1:
    st.title("Aplicación Para la Detección de Fraude Bancario - Paso 1: Carga y Limpieza de Datos")

    # Carga de archivo CSV
    uploaded_file = st.file_uploader("Selecciona un archivo CSV", type="csv")
    if uploaded_file is not None:
        # Leer datos originales solo una vez
        if 'original_df' not in st.session_state:
            df = pd.read_csv(uploaded_file)
            df = df.drop(["User_ID", "Timestamp","Transaction_ID"], axis=1)
            st.session_state.original_df = df
            st.session_state.df = df.copy()

        df_orig = st.session_state.original_df
        filas_orig, col_orig = df_orig.shape

        # Vista previa original
        st.subheader("Vista Previa de los Datos (Original)")
        st.dataframe(df_orig)

        # Gráficos de vista rápida (original)
        st.subheader("Boxplot por Variable")
        figures = boxplot(df_orig)
        for fig in figures:
            st.pyplot(fig)

        # Botón de limpiar datos
        if st.button("Limpiar Datos"):
            st.session_state.df = quitar_outliers(df_orig).reset_index(drop=True)
            st.success("Datos limpiados exitosamente.")

        # Mostrar vista tras limpieza si hubo cambios
        df_clean = st.session_state.df
        if not df_clean.equals(df_orig):
            filas_clean, col_clean = df_clean.shape
            st.subheader("Vista Previa tras Limpieza")
            st.dataframe(df_clean)
            st.write(f"Se quitaron {filas_orig - filas_clean} filas.")

            # Gráficos de vista rápida (limpios)
            st.subheader("Gráficos de Vista Rápida (Limpios)")
            figures_clean = boxplot(df_clean)
            for fig in figures_clean:
                st.pyplot(fig)

        # Botón para pasar al siguiente paso
        if st.button("Confirmar Datos y Continuar"):
            st.session_state.step = 2

# Paso 2: Selección de modelo y predicción
elif st.session_state.step == 2:
    st.title("Aplicación Para la Detección de Fraude Bancario - Paso 2: Predicción")
    df_clean = st.session_state.df
    df_clean = separar_variables(df_clean)

    # Mostrar datos que se usarán
    st.subheader("Datos Usados en Predicción")
    st.dataframe(df_clean)

    # Selección de modelo
    st.subheader("Selecciona el modelo de predicción")
    seleccion = st.selectbox("Modelos disponibles:", ["Regresión Logística", 'Red Neuronal'])

    # Botón de ejecutar predicción
    if st.button("Ejecutar Predicción"):
        if seleccion == 'Regresión Logística':
            df_rl = df_clean.copy()
            disp, metricas = linear_regresion(df_rl)
            st.subheader("Resultados de la Predicción")
            st.dataframe(metricas)
            fig, ax = plt.subplots()
            disp.plot(ax=ax)
            ax.set_title("Matriz de Confusión")
            st.pyplot(fig)
        else:
            with st.spinner("Entrenando el Modelo..."):
                df_rn = df_clean.copy() 
                epochs, loss, val_loss, metricas, disp = red_neuronal(df_rn)
                st.subheader("Métricas de Entrenamiento")
                st.dataframe(metricas)
                fig3,ax3 = plt.subplots()
                disp.plot(ax=ax3)
                ax3.set_title("Matriz de Confusión")
                st.pyplot(fig3)
                fig2, ax2 = plt.subplots()
                ax2.plot(epochs, loss, 'bo', label='Training loss')
                ax2.plot(epochs, val_loss, 'r', label='Validation loss')
                ax2.set_title('Training and Validation Loss')
                ax2.set_xlabel('Epochs')
                ax2.set_ylabel('Loss')
                ax2.legend()
                st.pyplot(fig2)
    # Botón para volver al paso anterior
    if st.button("Volver al Paso 1"):
        st.session_state.step = 1
