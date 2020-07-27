# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import pandas as pd
import os, getopt
from plotnine import *
pd.options.display.max_columns=100

#os.chdir("C:/Users/pabmo/Desktop/Decidata/Technalia_BigFash/Datos")

# PATHS
path_output = None
path_input = None
data_json = None

try:
    opts, args = getopt.getopt(sys.argv[1:],"i:o:d:",["ipath=","opath=","data="])
except getopt.GetoptError:
    print 'pendientes.py -ipath <path_input> -opath <path_output> -data data'
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-i", "--ipath"):
        path_input = arg
    elif opt in ("-o", "--opath"):
        path_output = arg
    elif opt in ("-d", "--data"):
        data_json = arg

if path_output is None:
    print "-o arg is required"
    sys.exit(2)

if path_input is None:
    print "-i arg is required"
    sys.exit(2)

if data_json is None:
    print "-d arg is required"
    sys.exit(2)

datos = pd.read_csv(path_input + "/datos.csv',sep=";")
datos.head()

datos.shape
datos.drop_duplicates("idanalytics",inplace=True)
datos.shape



############################
## sacamos Elementset y le damos forma de dataframe
############################
## ES: extraer:
#- action -> elementId
#- action -> name
#- action -> properties (esto es una lista de eventos)
#- action -> label (NO)
#- action -> category (NO)

def extraerInfo(datos, tipo):
    es = datos.loc[datos.analytics_event_type==tipo,:]
    es['analytics_data'] = es['analytics_data'].apply(lambda x : dict(eval(x)) )

    es2 = es['analytics_data'].apply(pd.Series )
    es2 = pd.concat([es2, es], axis = 1).drop('analytics_data', axis = 1)
    
    es3 = es2['action'].apply(pd.Series )
    es3 = pd.concat([es3, es2], axis = 1).drop('action', axis = 1)

    es4 = es3['properties'].apply(pd.Series )
    es4.columns = ["P0","P1","P2","P3"]
    es_final = pd.concat([es3, es4], axis = 1).drop('properties', axis = 1)

    esP0 = es_final['P0'].apply(pd.Series )
    esP0.columns = "P0_" + esP0.columns
    esP1 = es_final['P1'].apply(pd.Series )
    esP1.columns = "P1_" + esP1.columns
    esP2 = es_final['P2'].apply(pd.Series )
    esP2.columns = "P2_" + esP2.columns
    esP3 = es_final['P3'].apply(pd.Series )
    esP3.columns = "P3_" + esP3.columns

    esP0.head(3)
    esP1.head(3)
    esP2.head(3)
    esP3.head(3)

    es_final = pd.concat([es_final, esP0,esP1,esP2,esP3], axis = 1).drop(['P0','P1','P2','P3'], axis = 1)
    es_final.head()
    es_final.shape
    return es_final

es = extraerInfo(datos,"ELEMENTSET")
es.head(3)
pd.crosstab(es.P0_name,es.P0_value)

ms = extraerInfo(datos,"MATERIALSET")
ms.head(3)
pd.crosstab(ms.P0_name,ms.P0_value)



############################
## material set: name, P0_value, P2_value
############################
def formatoCorrecto(ms):
    #Hay que intercambiar P0 y P2 en algunos casos porque de acuerdo a la foto Orden a veces P0 almacena info de pendiente y a veces de enganche. Y lo mismo con P2.
    #Creamos variables Pendiente, Enganche, Pendiente_Color y Enganche_Color bien ordenadas
    ms["Pendiente"] = "No info"
    ms["Pendiente"][ms.P0_name.str.contains("Pendiente")] = ms["P0_name"][ms.P0_name.str.contains("Pendiente")]
    ms["Pendiente"][ms.P2_name.str.contains("Pendiente")] = ms["P2_name"][ms.P2_name.str.contains("Pendiente")]

    ms["Enganche"] = "No info"
    ms["Enganche"][ms.P0_name.str.contains("Enganche")] = ms["P0_name"][ms.P0_name.str.contains("Enganche")]
    ms["Enganche"][ms.P2_name.str.contains("Enganche")] = ms["P2_name"][ms.P2_name.str.contains("Enganche")]

    ms["Pendiente_Color"] = "No info"
    ms["Pendiente_Color"][ms.P0_name.str.contains("Pendiente")] = ms["P0_value"][ms.P0_name.str.contains("Pendiente")]
    ms["Pendiente_Color"][ms.P2_name.str.contains("Pendiente")] = ms["P2_value"][ms.P2_name.str.contains("Pendiente")]

    ms["Enganche_Color"] = "No info"
    ms["Enganche_Color"][ms.P0_name.str.contains("Enganche")] = ms["P0_value"][ms.P0_name.str.contains("Enganche")]
    ms["Enganche_Color"][ms.P2_name.str.contains("Enganche")] = ms["P2_value"][ms.P2_name.str.contains("Enganche")]

    print("Check de que pendientes y enganches no se han mezclado:")
    print(pd.crosstab(ms.Pendiente,ms.Enganche))

    return ms

ms = formatoCorrecto(ms)




############################
## Colores mas frecuentes por tipo de pendiente
############################

def generarExcelEImagen(ms):
    # Lo hacemos por Pendiente que están acumulados por familias. Si queremos más desglose usar name en vez de Pendiente
    writer = pd.ExcelWriter(path_output + '/Colores_de_Pendiente_y_Enganche_por_tipologia.xlsx')
    for i in ms.Pendiente.unique():
        print("Analizando pendiente: ",i)
        aux = ms.loc[ms.category=="MATERIALSET",:].loc[ms.Pendiente==i,:].loc[:,["Pendiente","Enganche","Pendiente_Color","Enganche_Color"]]
        print("Unicos pendientes \n",aux.Pendiente.value_counts())
        print("Unicos enganches \n",aux.Enganche.value_counts())
        aux = pd.crosstab(aux.Pendiente_Color,aux.Enganche_Color)
        aux.to_excel(writer,i)
        print("FINALIZADO\n-------------------------")
    writer.save()


    aux = ms.loc[ms.category=="MATERIALSET",:].loc[:,["Pendiente","Enganche","Pendiente_Color","Enganche_Color"]]
    a = ggplot(aes(x="Pendiente_Color",fill="Enganche_Color"),aux) + geom_bar()+coord_flip()+facet_wrap(("Pendiente"))
    a.save(path_output + "/Colores_de_Pendiente_y_Enganche_por_tipologia.pdf")

generarExcelEImagen(ms)





############################
## Retorno de predicciones
############################
aux = ms.loc[ms.category=="MATERIALSET",:].loc[:,["elementId","Pendiente","Enganche","Pendiente_Color","Enganche_Color"]]

def obtenerOutput(entrada, aux):
    import copy
    salida = copy.deepcopy(entrada)
    elementId = entrada["elementID"]
    sol = aux.loc[aux.elementId.astype(int)==elementId,:]
    if "properties" in entrada:
        tipo = entrada["properties"][0]["name"]
        color = entrada["properties"][0]["value"]
        if tipo == "Enganche":
            sol = sol.loc[sol.Enganche_Color == color,:]
            sol = sol.Pendiente_Color.value_counts().idxmax()
            salida["properties"].append({"name":"Pendiente","value":sol})
            return salida
        elif tipo == "Pendiente":
            sol = sol.loc[sol.Pendiente_Color == color,:]
            sol = sol.Enganche_Color.value_counts().idxmax()
            salida["properties"].append({"name":"Enganche","value":sol})
            return salida
        else:
            return "Estructura del input incorrecta"
    else:
        sol = sol.groupby(["Enganche_Color","Pendiente_Color"]).elementId.count().idxmax()
        salida["properties"] = [{"name":"Enganche","value":sol[0]},{"name":"Pendiente","value":sol[1]}]
        return salida

obtenerOutput(data_json, aux)

#1: Pendiente Beatle
#2: Pendiente Sfera
#3: Pendiente Quadrato
#4: Pendiente Letra A
#5: Pendiente Letra A Modelo y Pendiente Letra P
#6: Pendiente palabra leboh
#7: Pendiente palabra hakuna y pendiente palabra leboh modelo
#8: Pendiente Beatle modelo
#9: Pendiente Sfera Modelo
#10: Pendiente Quadrato Modelo
#11: Pendiente Letra P
#12: Pendiente Letra P Modelo
#13: Pendiente Palabra hakuna y Pendiente palabra leboh Modelo
#14: Pendiente palabra hakuna Modelo



