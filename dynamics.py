"""
Propiedades Dinámicas MD_numpy
=================================
    -Autocorrelaciones de velocidad
    -Coeficiente de Difusión D

@author: prangell
"""

import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
from scipy import stats    #Regresión de mínimos cuadrados

# %% LECTURA DE ARCHIVOS

nPart = 25
R = 1

files = []
dir_path = 'C:/Users/1812p/Desktop/UNIVERSIDAD/MUSCI/2º CUATRIMESTRE/Física Estadística Computacional/PRACTICAS/Fran Vega/pyMD-master/datos/'
for file in os.listdir(dir_path):
     if 'vxvy' in file:
         full_name = os.path.abspath(os.path.join(dir_path, file))
         files.append(full_name)
     if 'temp' in file:
         full_nameTemp = os.path.abspath(os.path.join(dir_path, file))
         dataTemp = pd.read_csv(full_nameTemp, delimiter = '\t')
dataV = pd.concat([pd.read_csv(file, delimiter = '\t') for file in files], ignore_index = True)

vx = dataV['vx']; vy = dataV['vy']
T = dataTemp['T']
T = np.mean(T)
nEst = len(vx) // nPart

# %% DISTRIBUCIÓN DE VELOCIDAD

vxMat = np.zeros((nPart, nEst))
vyMat = np.zeros((nPart, nEst))

for i in range(nEst):
    vxMat[:, i] = vx[i * nPart: (i + 1) * nPart]
    vyMat[:, i] = vy[i * nPart: (i + 1) * nPart]
    
# módulo de velocidad media
vMeanMod = np.zeros(len(vx))
for i in range(len(vx)):
    vMeanMod[i] = np.sqrt(vx[i]**2 + vy[i]**2)
    
# Función de Maxwell: distr. de módulo de velocidades media teórica
v = np.linspace(0,5,1000)
def maxwell(x,T):
    return (v/(0.5*T))*np.exp(-v**2/(2*0.5*T))

f=maxwell(v,T)

plt.figure(figsize=(6,6))
plt.hist(vMeanMod,bins=7,density=True, label = "Simulación")
plt.plot(v,f, label="Maxwell")
plt.title("Distribución velocidades")
plt.xlabel("Velocidad v")
plt.ylabel("Densidad de frecuencia")
plt.legend()
plt.show()
# plt.close()

#%% COEFICIENTE DE DIFUSIÓN
# from https://stackoverflow.com/questions/7489048/calculating-mean-squared-displacement-msd-with-matlab#:~:text=MSD%20is%20defined%20as%20MSD,particle%20over%20time%20interval%20t.

files = []
dir_path = 'C:/Users/1812p/Desktop/UNIVERSIDAD/MUSCI/2º CUATRIMESTRE/Física Estadística Computacional/PRACTICAS/Fran Vega/pyMD-master/datos/'
for file in os.listdir(dir_path):
      if 'xy' in file:
          full_name = os.path.abspath(os.path.join(dir_path, file))
          files.append(full_name)
 
x = pd.DataFrame(); y = pd.DataFrame()
for file in files:
    df = pd.read_csv(file, delimiter = '\t', skiprows = 1)
    x = pd.concat([x, df.iloc[:,0].to_frame().T], axis = 0, ignore_index = True)
    y = pd.concat([y, df.iloc[:,0].to_frame().T], axis = 0, ignore_index = True)

# x, y son dataframes con las posiciones donde las filas indican en qué paso temporal, columnas la partícula

deltaT = int(np.floor(len(x)))      #núm de pasos temporales simulados
msd = np.zeros((deltaT-1, nPart))  
    
for dt in range(1,deltaT): 
    deltaX = np.array(x[dt:]) - np.array(x[:-dt])
    deltaY = np.array(y[dt:]) - np.array(y[:-dt])
    disp2 = np.mean(deltaX**2+deltaY**2)
    msd[dt-1] = disp2
    
MSD = np.mean(msd, axis = 1)

plt.figure(figsize = (6,6))
plt.plot(MSD, '.')
plt.xlabel('Tiempo en unidades de tiempo de colisión')
plt.ylabel('MSD')
plt.show()

# Mínimos cuadrados para obtener D
MSD_regression = MSD[10:120]            #Despreciamos la parte balística y donde se empieza a estabilizar
t = np.arange(10, 120)

res = stats.linregress(t,MSD_regression)


print('MÍNIMOS CUADRADOS')
print('=======================')
print('Pendiente = ', res.slope) 
print('Ord. origen = ', res.intercept)

# Distribución t de Student 
# p - probabilidad, df - grados de libertad
tinv = lambda p, df: abs(stats.t.ppf(p/2, df))
ts = tinv(0.05, len(x)-2)

print('ERRORES')
print('=======================')
print(f'R^2: {res.rvalue**2:.6f}')
print(f"Pendiente (95%): {res.slope:.6f} +/- {ts*res.stderr:.6f}")
print(f"Ord. origen (95%): {res.intercept:.6f}"
      f" +/- {ts*res.intercept_stderr:.6f}")

tt = np.linspace(10, 120, 500)
yy = res.intercept + res.slope*tt

plt.figure(figsize = (6,6))
plt.plot(t, MSD_regression, '.', label = 'Datos')
plt.plot(tt, yy, '-r',lw = 2,  label = f'Ajuste por mínimos cuadrados MSD = {res.slope:.2}t {res.intercept:.2}')
plt.xlabel('Tiempo en unidades de tiempo de colisión')
plt.ylabel('MSD')
plt.legend()
plt.show()

