
# CARGAR LIBRERIAS
import sys
import os, getopt
import tempfile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import os.path
import pandas as pd
import glob
import xlrd
from datetime import datetime, timedelta
import math
import matplotlib as matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import pylab as pl
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pickle
import joblib
import tempfile
import calendar
import pandas_profiling
from pandas_profiling import ProfileReport

# WEB SCRAPPING (FALTA EL SCRIPT)
import requests
import urllib.request
import time
from bs4 import BeautifulSoup

# DEEP LEARNING
#from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from functools import partial
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()
stop_words = set(stopwords.words('english'))
import spacy
import itertools as it
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import cifar10
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense
from keras.layers import Dropout
from functools import partial


# CONFIGURACION DE LAS FIGURAS
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# HACER QUE PANDAS MUESTRE TODAS LAS COLUMNAS
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# DESACTIVAR Warnings
pd.options.mode.chained_assignment = None
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# PATHS
path_output = None
path_input = None
file_input = None

try:
    opts, args = getopt.getopt(sys.argv[1:],"i:o:f:",["ipath=","opath=","file="])
except getopt.GetoptError:
    print('restaurantes.py -ipath <path_input> -opath <path_output> -f <file_input> ')
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-i", "--ipath"):
        path_input = arg
    elif opt in ("-o", "--opath"):
        path_output = arg
    elif opt in ("-f", "--file"):
        file_input = arg

if path_output is None:
    print('-o arg is required')
    sys.exit(2)

if path_input is None:
    print ('-i arg is required')
    sys.exit(2)

if file_input is None:
    print ('-f arg is required')
    sys.exit(2)

def process_data_1():
    # CARGAR DATOS EXCEL (FALTA AGREGAR CSV)
    xls = pd.ExcelFile(file_input)
    df = pd.read_excel(xls, 'datos')
    # TRANSFORMAR DATE EN datetime64[ns]
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y/%m/%d')
    # ELIMINAR DUPLICADOS
    df = df.drop_duplicates()
    # CAMBIAR EL NOMBRE DE LA COLUMNA Orders POR Today_Orders
    df = df.rename(columns={'orders': 'today_orders'})
    # AGREGAR FECHAS QUE FALTAN
    r = pd.date_range(start=df.date.min(), end=df.date.max())
    df = df.set_index('date').reindex(r).rename_axis('date').reset_index()
    # AGREGAR DIA DE LA SEMANA
    df['day_week'] = df['date'].dt.day_name()
    # SUSTITUIR DESCRIPCION 1 - 7
    df['day_week'] = df['day_week'].map({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7})
    # AGREGAR NUMERO DEL MES
    df['month'] = df['date'].dt.month
    df_1 = df
    return df_1

df_1 = process_data_1()

def process_data_2():
    # CARGAR DATOS EXCEL (FALTA AGREGAR CSV)
    xls = pd.ExcelFile(path_input+'/weather.xlsx')
    df = pd.read_excel(xls, 'Hoja1')
    # TRANSFORMAR DATE EN datetime64[ns]
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y/%m/%d')
    df_2 = df
    return df_2

df_2 = process_data_2()

def process_data_3():
    # CARGAR DATOS EXCEL (FALTA AGREGAR CSV)
    xls = pd.ExcelFile(path_input+'/holiday.xlsx')
    df = pd.read_excel(xls, 'Hoja1')
    # TRANSFORMAR DATE EN datetime64[ns]
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y/%m/%d')
    # SUSTITUIR DESCRIPCION POR 1 & 0
    df['holiday'] = df['holiday'].map({'Yes': 1, 'No': 0})
    df_3 = df
    return df_3

df_3 = process_data_3()

# COMBINAR DATASETS (df_1, df_2 & df_3)
original_1 = pd.merge(df_1, df_2, how="inner")
original = pd.merge(original_1, df_3, how="inner")

# VARIABLE OBJETIVO
# CREAR VARIABLE Orders+3
objetivo_1 = original
# CREAR COLUMNA CON FECHA MAS 3 DIAS
objetivo_1['nueva_fecha'] = original['date'] - timedelta(days=3)
# AGREGAR LOS PEDIDOS DE FECHA MAS 3 DIAS
objetivo_1 = objetivo_1.loc[:,["nueva_fecha", "today_orders", 'month', "day_week", "temperature", "precipitation", "condition", "holiday"]]
# CAMBIAR NOMBRES
objetivo_1 = objetivo_1.rename(columns={'today_orders': 'orders_N3', 'month': 'month_N3', 'day_week': 'day_week_N3', 'temperature': 'temperature_N3', 'precipitation': 'precipitation_N3', 'condition': 'condition_N3', 'holiday': 'holiday_N3'})

# CREAR VARIABLE Orders-4
objetivo_2 = original
# CREAR COLUMNA CON FECHA MENOS 4 DIAS
objetivo_2['nueva_fecha'] = original['date'] + timedelta(days=4)
# AGREGAR LOS PEDIDOS DE FECHA MENOS 4 DIAS
objetivo_2 = objetivo_2.loc[:,["today_orders", "nueva_fecha"]]
# CAMBIAR NOMBRES
objetivo_2 = objetivo_2.rename(columns={'today_orders': 'orders_N4'})
# ORDENAR COLUMNAS
objetivo_2 = objetivo_2[['nueva_fecha', 'orders_N4']]


# CREAR MASTER
premaster1 = pd.merge(original, objetivo_1, how="inner", left_on="date", right_on="nueva_fecha")
# print(premaster1.head(20))
# ELIMINAR COLUMNAS INNECESARIAS
del premaster1['day_week']
del premaster1['month']
del premaster1['temperature']
del premaster1['precipitation']
del premaster1['condition']
del premaster1['holiday']
del premaster1['nueva_fecha_x']
del premaster1['nueva_fecha_y']

premaster2 = pd.merge(premaster1, objetivo_2, how="inner", left_on="date", right_on="nueva_fecha")
# ELIMINAR COLUMNAS INNECESARIAS
del premaster2['nueva_fecha']
# ORDENAR COLUMNAS
premaster2 = premaster2[['date', 'orders_N4', 'today_orders', 'orders_N3', 'month_N3', 'day_week_N3', 'temperature_N3', 'precipitation_N3', 'condition_N3', 'holiday_N3']]
# REVISION
# print(premaster2.head(20))

# MASTER
master = premaster2
# REVISAR SI HAY VALORES NULOS
# print(master.isnull().values.sum())
# ELIMINAR ZEROs y NaNs
master = master[master['today_orders'].notna()]
master = master[master['today_orders'] != 0]
master = master[master['orders_N3'].notna()]
master = master[master['orders_N3'] != 0]
master = master[master['orders_N4'].notna()]
master = master[master['orders_N4'] != 0]
# REVISAR SI HAY VALORES NULOS
# print(master.isnull().values.sum())
# # ELIMINAR OUTLIERS
# master = master[master['today_orders'] < 27]
# master = master[master['orders_N4'] < 27]
# master = master[master['orders_N3'] < 27]
# REVISION
# print(master)
# CREAR DATASET EN FORMATO CSV
master.to_csv(path_input+'/master_prueba.csv')


# PREPARAR LAS ENTRADAS PARA LOS MODELOS DE ML
restaurant_test = pd.read_csv(path_input+'/master_sin_outliers.csv', sep=",")
# VARIABLES QUE INFLUYEN
model = restaurant_test[['orders_N4', 'today_orders', 'month_N3', 'day_week_N3', 'temperature_N3', 'precipitation_N3', 'condition_N3', 'holiday_N3']]
# VARIABLE A PREDECIR
target = restaurant_test[['orders_N3']]
# TRAIN & TEST
X_train, X_test, y_train, y_test = train_test_split(model, target, test_size=0.2, random_state=42)

# VISUALIZAR EL TRAIN & TEST
# print("X_train: "+X_train.shape.__str__())
# print("X_test: "+X_test.shape.__str__())
# print("y_train: "+y_train.shape.__str__())
# print("y_test: "+y_test.shape.__str__()+ '\n')

# RFR
# # ENCONTRAR LOS MEJORES HIPERPARAMETROS
# # DEFINIR EL RANGO DE LOS HIPERPARAMETROS
# param_grid_RFR = {'bootstrap': [True], 'max_depth': [38, 40, 42, 44], 'min_samples_leaf': [1, 2, 3], 'min_samples_split': [2, 4, 5], 'n_estimators': [94, 96, 98, 100], 'max_samples': [42, 44, 46, 48]}
# # AUTOMATIZAR LA BUSQUEDA CON GridSearchCV
# model_RFR = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid_RFR, cv=5)
# # VALIDACION CRUZADA
# modelfit_RFR = model_RFR.fit(X_train, y_train)
# # MEJOR PRECISION
# RFR_accuracy = modelfit_RFR.best_score_
# print(RFR_accuracy)
# # MEJORES HIPERPARAMETROS
# RFR_parameters = modelfit_RFR.best_params_
# print(RFR_parameters)

# CARGAR LOS MEJORES HIPERPARAMETROS EN EL ALGORITMO
rfr = RandomForestRegressor(bootstrap=True, max_depth=40, max_samples=44, min_samples_leaf=1, min_samples_split=2, n_estimators=98)
# FIT (ENTRENAMIENTO)
rfr.fit(X_train, y_train)
# PREDICCIONES DE NUEVOS RESULTADOS
pred_rfr = rfr.predict(X_test)
# COMPARACION
y_test['Predicciones'] = pred_rfr
# print(y_test.head(20))
# REDONDEAR DECIMALES
Comparaciones = y_test.round(1)


Comparaciones_1 = Comparaciones
Comparaciones_1['RFR_Error'] = (Comparaciones['orders_N3'] - Comparaciones['Predicciones']).abs().round(2)
Comparaciones_1['RFR_Error_%'] = ((Comparaciones['orders_N3'] - Comparaciones['Predicciones']) / Comparaciones['Predicciones']).abs().round(2)
# ORDENAR COLUMNAS
Comparaciones_1 = Comparaciones_1[['orders_N3', 'Predicciones', 'RFR_Error', 'RFR_Error_%']]
# print(Comparaciones_1)

mse = mean_squared_error(Comparaciones['orders_N3'], Comparaciones['Predicciones']).round(2)
# print("MSE RFR: "+mse.__str__())
mae = mean_absolute_error(Comparaciones['orders_N3'], Comparaciones['Predicciones']).round(2)
# print("MAE RFR: "+mae.__str__()+ '\n')


# RFR
fig, ax = plt.subplots()
ax.scatter(x="Predicciones", y="orders_N3", data=Comparaciones, c='black')
ax.set_ylim([0, 40])
ax.set_xlim([0, 40])
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.title('RFR')
plt.savefig(path_output+'/Grafico_1')


# SAVE MODEL
restaurant_test = pd.read_csv(path_input+'/master_sin_outliers.csv', sep=",")

model = restaurant_test[['orders_N4', 'today_orders', 'month_N3', 'day_week_N3', 'temperature_N3', 'precipitation_N3', 'condition_N3', 'holiday_N3']]
target = restaurant_test[['orders_N3']]
X_train, X_test, y_train, y_test = train_test_split(model, target, test_size=0.2, random_state=42)

rfr = make_pipeline(RandomForestRegressor(bootstrap=True, max_depth=40, max_samples=44, min_samples_leaf=1, min_samples_split=2, n_estimators=98))
rfr.fit(X_train, y_train)

with open (path_input+'/modelo_pedidos', 'wb') as f:
    pickle.dump(rfr, f)


# LOAD MODEL
new_data = pd.read_csv(path_input+'/master.csv')

with open (path_input+'/modelo_pedidos', 'rb') as f:
    rfr = pickle.load(f)

rfr

restaurant_test = pd.read_csv(path_input+'/master.csv')

model = restaurant_test[['orders_N4', 'today_orders', 'month_N3', 'day_week_N3', 'temperature_N3', 'precipitation_N3', 'condition_N3', 'holiday_N3']]
target = restaurant_test[['orders_N3']]
X_train, X_test, y_train, y_test = train_test_split(model, target, test_size=0.2, random_state=42)

X_train.shape

print('')
print("PREDICCIONES:")
print('')
print(rfr.predict(X_train).round(1))

# RESULTADO
resultado = rfr.predict(X_train).round(1)
# CREAR CSV
#resultado.to_csv(path_output+'/predicciones.csv')
prediction = pd.DataFrame(resultado, columns=['predicciones']).to_csv(path_output+'/predicciones.csv')


# GRAFICOS

# CARGAR EL DATASET
dataset = pd.read_csv(path_input+'/original.csv', sep=",")
del dataset['Unnamed: 0']
dataset = dataset[dataset['today_orders'].notna()]
dataset = dataset[dataset['today_orders'] != 0]
dataset['date'] = pd.to_datetime(dataset['date'].astype(str), format='%Y/%m/%d')

# # E X P L O R A T O R I O
# prof = PropathReport(dataset)
# print(prof)
# prof.to_file(output_file='exploratory_analysis.html')

# # D E S C R I P T I V O
# # MOSTRAR VALORES DEL DATASET
# print(dataset.head())
# # MOSTRAR INFO DE LAS VARIABLES DEL DATASET
# print(dataset.info())
# # MOSTRAR MEDIDAS DESCRIPTIVAS
# print(dataset.describe())
# # MOSTRAR DIMENSIONES DEL DATASET
# print(dataset.shape)
# # REVISAR SI HAY VALORES NULOS
# print("Valores Nulos: "+dataset.isnull().values.sum().__str__())
# # HISTOGRAMA (no aporta mucha info)
# dataset['today_orders'].hist()

# BOXPLOT
# CON OUTLIERS
sns.set(style="whitegrid")
plt.figure(figsize=(10, 8))
ax = sns.boxplot(x='today_orders', data=dataset, orient="v")
plt.title('Visualización de outliers')
plt.savefig(path_output+'/Grafico_2')
# SIN OUTLIERS
# ELIMINAR OUTLIERS
dataset_so = dataset[dataset['today_orders'] < 27]
sns.set(style="whitegrid")
plt.figure(figsize=(10, 8))
ax = sns.boxplot(x='today_orders', data=dataset_so, orient="v")
plt.title('Visualización de outliers')
plt.savefig(path_output+'/Grafico_3')

# CORRELACION
matrix_corr = dataset_so.corr()
# print(matrix_corr)
sns.heatmap(matrix_corr, annot=True)
plt.yticks(rotation=360)
plt.title('Correlación de variables')
plt.savefig(path_output+'/Grafico_4')


# A N O V A
# CREAR UN DATASET SEMANAS DEL AÑO  Y Pedidos x Dias de la semana
# SELECCIONAR COLUMNAS PARA ANOVA
df_anova = dataset.copy()
df_anova = df_anova[["date", "today_orders", "day_week"]]
# df_anova = df_anova[df_anova['today_orders'].notna()]
df_anova
# PIVOT TABLE
df_anova1 = df_anova.pivot_table('today_orders', ['date'], 'day_week')
# anova = anova.set_index('day_week')
# anova1 = anova.droplevel("date")
# anova1.reset_index()
# print(df_anova1)
# print(df_anova1.info())
df_anova1.to_csv(path_input+'/df_anova1.csv')
# BOXPLOT POR DIAS DE SEMANA
df_anova.boxplot('today_orders', by='day_week', figsize=(12, 8))
mod = ols('today_orders ~ day_week', data=df_anova).fit()
plt.savefig(path_output+'/Grafico_5')
aov_table = sm.stats.anova_lm(mod) # , typ=2)
# print(aov_table)

# T U K E Y
# pairwise_tukeyhsd(df_anova1, day_week)


# G R A F I C O S

# MASK DE PEDIDOS POR FECHA
start_date = '2016-01-01'
end_date = '2019-12-31'
mask = (dataset['date'] >= start_date) & (dataset['date'] <= end_date)




# POR DIA DE LA SEMANA
# AGREGAR NUMERO DEL AÑO
dataset_year = dataset
# FILTRO DE TIEMPO
dataset_year = dataset.loc[mask]
# EL GROUPBY TRANSFORMA EL DF EN SERIE
dataset_year_2 = dataset_year.sort_values(by="date").groupby('day_week')['today_orders'].sum()
# print(dataset_year_2)
# TRANSFORMA SERIE EN DF
dataset_year_3 = pd.DataFrame({'day_week':dataset_year_2.index, 'orders':dataset_year_2.values})
# print(dataset_year_3)
sns.catplot(x='day_week', y='orders', data=dataset_year_3, kind="bar", aspect=3)
plt.title('Distribución de pedidos por día de la semana 2017 - 2018')
plt.xlabel('day_week')
plt.ylabel('orders')
plt.savefig(path_output+'/Grafico_6')


# COMPARACION PEDIDOS POR DIA DE SEMANA 2017-2018
start_date = '2016-01-01'
end_date = '2019-12-31'
mask = (dataset['date'] >= start_date) & (dataset['date'] <= end_date)
dataset_year = dataset.loc[mask]
dataset_year['year'] = dataset_year['date'].dt.year
#
sns.set_style("white")
fig, ax = plt.subplots(figsize=(4,4))
sns.catplot(x="day_week", y="today_orders", hue="year", kind="bar", data=dataset_year, ax=ax, ci=None)
ax.legend(['2017', '2018'], facecolor='w')
plt.title('Comparación de pedidos por día de semana 2017 - 2018')
plt.savefig(path_output+'/Grafico_7')

# HEATMAP
# PROMEDIO DE PEDIDOS POR DIAS Y MESES
dataset_matrix = pd.pivot_table(dataset_year, values='today_orders', index=['day_week'], columns='month')
# print(dataset_matrix.head(50))
a = sns.heatmap(dataset_matrix, annot=True)
a.vlines([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], *a.get_xlim())
# a.hlines([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], *a.get_xlim())
plt.title('Promedio de pedidos por días y meses 2018')
plt.xlabel('month')
plt.ylabel('day_week')
plt.savefig(path_output+'/Grafico_8')


# # PROMEDIO DE PEDIDOS POR DIAS Y SEMANAS
# dataset_week = dataset.copy()
# dataset_week1 = dataset_week.loc[mask]
# dataset_week1['week_number'] = dataset_week1['date'].dt.week
# del dataset_week1['temperature']
# del dataset_week1['precipitation']
# del dataset_week1['condition']
# del dataset_week1['holiday']
# del dataset_week1['month']
# del dataset_week1['date']
# dataset_week1 = dataset_week1[['week_number', 'day_week', 'today_orders']]
# # print(dataset_week1)
# dataset_week1.to_csv('../Data/dataset_week1.csv')
# full_matrix = pd.pivot_table(dataset_week, values='today_orders', index=['day_week'], columns='week_number')
# # print(full_matrix)
# a = sns.heatmap(full_matrix, annot=True)
# plt.title('Pedidos por días y semanas 2017')
# plt.xlabel('week_number')
# plt.ylabel('day_week')




# GRAFICO DE LINEAS POR DIA
# FILTRO DE TIEMPO
start_date = '2017-01-01'
end_date = '2017-12-31'
mask = (dataset['date'] >= start_date) & (dataset['date'] <= end_date)
dataset_1 = dataset.loc[mask]
# EL GROUPBY TRANSFORMA EL DF EN SERIE
dataset_1 = dataset_1.sort_values(by="date").groupby('month')['today_orders'].sum()
# print(dataset_1)
# TRANSFORMA SERIE EN DF
dataset_11 = pd.DataFrame({'date':dataset_1.index, 'orders':dataset_1.values})
# print(dataset_11)
# VS
# FILTRO DE TIEMPO
start_date = '2018-01-01'
end_date = '2018-12-31'
mask = (dataset['date'] >= start_date) & (dataset['date'] <= end_date)
dataset_2 = dataset.loc[mask]
# EL GROUPBY TRANSFORMA EL DF EN SERIE
dataset_2 = dataset_2.sort_values(by="date").groupby('month')['today_orders'].sum()
# print(dataset_1)
# TRANSFORMA SERIE EN DF
dataset_22 = pd.DataFrame({'date':dataset_2.index, 'orders':dataset_2.values})
# print(dataset_22)
# GRAFICAR (UTILIZAR MASK PARA CAMBIAR LOS PERIODOS)
sns.set_style("white")
fig, ax = plt.subplots(figsize=(4,4))
sns.lineplot(x='date', y='orders', data=dataset_11, color='r', ax=ax, ci=None)
sns.lineplot(x='date', y='orders', data=dataset_22, color='b', ax=ax, ci=None)
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_major_locator(plt.MultipleLocator(50))
ax.set_xlim(0,12)
ax.set_ylim(0,500)
ax.legend(['2017', '2018'], facecolor='w')
plt.title('Pedidos por mes 2017 - 2018')
plt.xlabel('date')
plt.ylabel('orders')
plt.savefig(path_output+'/Grafico_9')


print('')
print("...:::| ENDED PROCESS |:::...")





