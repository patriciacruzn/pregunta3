# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 15:32:09 2021

@author: Desktop
"""
#RELAIZAR TRES ALGORITMOS
from sklearn import preprocessing
import numpy as np
import pandas as pd

datos=pd.read_csv("GlobalLandTemperaturesByState1.csv")
datos1=np.array(datos['AverageTemperature'])
datos2=np.array(datos['AverageTemperatureUncertainty'])
datos0=np.array(datos['dt'])
datos3=np.array(datos['State'])
datos4=np.array(datos['Country'])
datos33=pd.get_dummies(datos['State'], prefix='State')
datos44=pd.get_dummies(datos['Country'], prefix="country")
print("*******************************************************************************")
print("ORIGINAL")
print(datos[:20])
print("*******************************************************************************")
print("PAIS")
print(datos44[:20])
print("*******************************************************************************")
print("ESTADO")
print(datos33[:20])
datos=pd.concat([datos,datos33],axis=1)
datos=pd.concat([datos,datos44],axis=1)
datossolo=datos.drop(['dt','AverageTemperature','AverageTemperatureUncertainty','State','Country'],axis=1)
print(datossolo[:20])
print("*******************************************************************************")
print("TEMPERATURA MEDIA")
print(datos1[:20])
print("PRIMERO")
Aprepro=preprocessing.scale(datos1[:20])
print(Aprepro)
print("SEGUNDO")
Aprepro1=preprocessing.maxabs_scale(datos1[:20])
print(Aprepro1)
print("TERCERO")
Aprepro2=preprocessing.minmax_scale(datos1[:20])
print(Aprepro2)
