import streamlit as st
import pandas as pd
import numpy as np

# Título de la aplicación
st.title('Hola, Mundo con Streamlit!')

# Subtítulo
st.write("Este es un ejemplo de una aplicación simple en Streamlit.")

# Crear un dataframe
df = pd.DataFrame({
    'A': np.random.randn(10),
    'B': np.random.randn(10),
    'C': np.random.randn(10),
})

# Mostrar el dataframe en la aplicación
st.write("Aquí hay un dataframe aleatorio:")
st.dataframe(df)

# Crear un gráfico
st.write("Y aquí hay un gráfico:")
st.line_chart(df)

# Entrada de texto
user_input = st.text_input("Escribe algo aquí:")
st.write("Has escrito:", user_input)

# Slider
x = st.slider('Selecciona un valor:', 0, 100, 50)
st.write('El valor seleccionado es:', x)

# Selector de fecha
import datetime
d = st.date_input("Selecciona una fecha:", datetime.date(2023, 1, 1))
st.write('La fecha seleccionada es:', d)

# Botón
if st.button('Saludar'):
    st.write('¡Hola, mundo!')

# Checkbox
agree = st.checkbox('Estoy de acuerdo')
if agree:
    st.write('¡Genial!')