"""
Propiedades Estructurales MD_numpy
=================================
    -Distribución radial 
    -Teselación de Voronoi

@author: prangell
"""

import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay,Voronoi, voronoi_plot_2d
import imageio        # para pintar gif


# %% Funciones

def delanuay_gif(name):
    
    global nEst, nPart, R
    # genera gif con la evolución de la  triangulación de Delanuay

    frames = []
    for i in range(nEst):
        x = xMat[:, i]
        y = yMat[:, i]
        state = np.column_stack((x, y))
        triang = Delaunay(state)
        triang = triang.simplices

        fig, axs = plt.subplots(figsize = [6,6])
        axs.triplot(x, y, triang)
        for j in range(nPart):
            circle = plt.Circle((x[j], y[j]), R, color = 'b')
            axs.add_artist(circle)
        axs.set_xlabel('x')
        axs.set_ylabel('y')
        axs.set_title(f'Triang. Delanuay estado {i+1}')
        plt.savefig(f'img/delaunay/img_{i}.png',
                        transparent=False,
                        facecolor='white')
        plt.close()
        image = imageio.v2.imread(f'img/delaunay/img_{i}.png')
        frames.append(image)
    
    # Combinamos todos los frames en un GIF
    imageio.mimsave(name+'.gif',
                    frames, fps = 5, loop = 1)

def voronoi_gif(name):
    
    # genera gif con la evolución de la teselación de Voronoi
    global nEst, nPart, R

    frames = []
    for i in range(nEst):
        x = xMat[:, i]
        y = yMat[:, i]
        state = np.column_stack((x, y))
        vor = Voronoi(state)
        
        fig, axs = plt.subplots(figsize=[6, 6])
        voronoi_plot_2d(vor, ax = axs)
        for j in range(nPart):
            circle = plt.Circle((x[j], y[j]), R, color='b')
            axs.add_artist(circle)
        axs.set_xlabel('x')
        axs.set_ylabel('y')
        axs.set_title(f'Teselación Voronoi estado {i+1}')
        plt.savefig(f'img/voronoi/img_{i}.png',
                    transparent=False,
                    facecolor='white')
        plt.close()
        image = imageio.v2.imread(f'img/voronoi/img_{i}.png')
        frames.append(image)

    # Combinamos todos los frames en un GIF
    imageio.mimsave(name+'.gif',
                    frames, fps=5, loop=1)


# %% LECTURA DE ARCHIVOS

nPart = 25
R = 1

files = []
dir_path = 'C:/Users/1812p/Desktop/UNIVERSIDAD/MUSCI/2º CUATRIMESTRE/Física Estadística Computacional/PRACTICAS/Fran Vega/pyMD-master/datos/'
for file in os.listdir(dir_path):
     if 'xy' in file:
         full_name = os.path.abspath(os.path.join(dir_path, file))
         files.append(full_name)
         
data = pd.concat([pd.read_csv(file, delimiter = '\t') for file in files], ignore_index = True)

x = data['x']; y = data['y']
nEst = len(x) // nPart

# %% DISTRIBUCIÓN RADIAL

xMat = np.zeros((nPart, nEst))
yMat = np.zeros((nPart, nEst))

for i in range(nEst):
    xMat[:, i] = x[i * nPart: (i + 1) * nPart]
    yMat[:, i] = y[i * nPart: (i + 1) * nPart]
    
# Función de distribución radial de cada partícula con el resto para cada paso temporal
dist = np.zeros((nPart * (nPart - 1) // 2, nEst))


k = 0
for i in range(nPart):
    for j in range(i+1,nPart):
        dist[k] = np.sqrt((xMat[i] - xMat[j])**2 + (yMat[i] - yMat[j])**2)
        k += 1
        
dist = dist.flatten()

plt.figure(figsize=(6,6))
plt.hist(dist,bins=25,density=True)
plt.title("Distribución radial g(r)")
plt.xlabel("Distancia r")
plt.ylabel("Densidad de frecuencia")
plt.show()
plt.close()

# %% TRIANGULACIÓN DE DELANUAY 

# En nt = 1 ( antes de la primera colisión)
x0=xMat[:,1]; y0=yMat[:,1]
firstState = np.column_stack((x0, y0)) 
triang = Delaunay(firstState)                   # Triangulación de Delanuay

triang = triang.simplices                       # Triángulos

fig, axs = plt.subplots(figsize = [6,6])

for i in range(nPart):
    circle = plt.Circle((x0[i], y0[i]), R, color = 'b')
    axs.add_artist(circle)
    
axs.set_xlabel('x')
axs.set_ylabel('y')
axs.set_title('Teselación de Voronoi estado inicial')
plt.triplot(x0, y0, triang)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Triangulación de Delaunay estado inicial')
plt.show()
plt.close()

# Gif con las triangulaciones tras cada colisión
delanuay_gif('delanuay_Part25_col10')

# %% TESELACIÓN DE VORONOI

# En nt = 1 (antes de la primera colisión)
vor = Voronoi(firstState)

fig, axs = plt.subplots(figsize = [6,6])
voronoi_plot_2d(vor, ax = axs)
for i in range(nPart):
    circle = plt.Circle((x0[i], y0[i]), R, color = 'b')
    axs.add_artist(circle)
    
axs.set_xlabel('x')
axs.set_ylabel('y')
axs.set_title('Teselación de Voronoi estado inicial')
plt.show()
plt.close()

# Gif con las teselaciones tras cada colisión
voronoi_gif('voronoi_Part25_col10')