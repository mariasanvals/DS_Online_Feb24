# Importar librerías necesarias
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el dataset 'tips' de Seaborn
tips = sns.load_dataset('tips')

# Mostrar las primeras filas del DataFrame
print(tips.head())

# Describir el DataFrame
print(tips.describe())

# Crear algunas visualizaciones

# Distribución del total de la factura
plt.figure(figsize=(10, 6))
sns.histplot(tips['total_bill'], kde=True)
plt.title('Distribución del Total de la Factura')
plt.xlabel('Total de la Factura')
plt.ylabel('Frecuencia')
plt.show()


# Boxplot de la propina por día
plt.figure(figsize=(10, 6))
sns.boxplot(x='day', y='tip', data=tips)
plt.title('Boxplot de la Propina por Día')
plt.xlabel('Día de la Semana')
plt.ylabel('Propina')
plt.show()

# Scatter plot del total de la factura vs la propina
plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_bill', y='tip', data=tips, hue='time', style='time', size='size', sizes=(20, 200))
plt.title('Total de la Factura vs Propina')
plt.xlabel('Total de la Factura')
plt.ylabel('Propina')
plt.show()

print(tips.head(50))
# Heatmap de la correlación entre variables numéricas
plt.figure(figsize=(10, 6))
sns.heatmap(tips.drop(['smoker','sex','day','time'],axis=1).corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Heatmap de la Correlación entre Variables')
plt.show()