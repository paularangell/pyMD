# DINAMICA MOLECULAR EN python, DISCOS DUROS    #####
# Fisica Estadistica, UEx, en Roma, mayo 2015   #####
                                                #####
# VERSION 1.0                                   #####
                                                #####
# en modo terminal, ejecutar con:               #####
                                                #####
# 'python MD.py'                                #####
                                                #####
#####################################################


# importa librerias necesarias: "math", "numpy, "random", "bisect", "operator" y "system"
# usualmente todas ya vienen instaladas en el nucleo python excepto "numpy" 
# hay una version en git/pyMD
import math
import numpy as np
import random
import bisect # libreria de ordenacion de listas (para lista de cols.)
from operator import itemgetter, attrgetter
# import matplotlib # esta es para usar graficos python. a implementar en nuevas versiones
import os
import pandas as pd

import matplotlib.pyplot as plt
from progress.bar import IncrementalBar as Bar
import imageio

# %% PARÁMETROS

R=1.                # radio de las particulas
npart = 25          # numero de particulas para la simulacion
nt = 10 * npart     # numero de pasos temporales (cols. x particula)
alfa = 1.0          # coef. de restitucion

tol = 1.0e-20       # parametro de control para evitar solapacion de parts. por error numerico 
ncp=1.0*nt/npart

utermo = 1          # iteraciones entre snapshots que sale (1 para que salgan todas)

# RESUMEN 
print(" ")
print("SIMULACION MD")
print("alfa= ", alfa)
print("cols/part (total): ", ncp)
#print("cols/part entre snapshots: ", ncp)
print("Iteraciones entre snapshots:  ", utermo)
print("Núm. de archivos:  ", nt/utermo)

# %% INICIALIZACIONES

#   inicializa listas temporales de T y a2 
temp = np.zeros(nt+1)
a2 = np.zeros(nt+1)

#inicializa listas relacionadas con las colisiones
listacol = []
listacol_orden = []
ij = []

# inicializa el tiempo
t = 0.
dt = 0.
it = 0

#   inicializa el generador aleatorio. cada vez que se lanza la simulacion usa una semilla aleatoria
# es decir, ejecuciones consecutivas hacen simulaciones estadisticamente diferentes (replicas)
# si no se quiere esta propiedad, escribir: random.seed(1)
random.seed()


# %% FUNCIONES

def initialize_xv_arrays():
    # inicializa arrays de pos, vel
    
    global x, y, vx, vy
    #   inicializa arrays de posiciones
    x = np.zeros(npart)
    y = np.zeros(npart)
    #   inicializa arrays de velocidades    
    vx = np.zeros(npart)
    vy = np.zeros(npart)
    
    
def initialize_random():
    global x, y, vx, vy

    initialize_xv_arrays()
    
    # colocacion de las particulas
    x[0]=random.uniform(-LXR, LXR)
    y[0]=random.uniform(-LYR, LYR)

    # condicion de solapamiento
    for i in range(1,npart):
       dr=False
       while dr==False:
           x[i]=random.uniform(-LXR, LXR)
           y[i]=random.uniform(-LYR, LYR)
           # condicion de no solapamiento con las pos. ya generadas
           for j in range(0,i):
               dr=((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j])>4*R*R)
               if dr==False:
                  break

    
def initialize_square():    
    global x, y, LX, LY, npart

    NL = int(round( np.sqrt(npart) ))
    npart = NL**2 # correct no. of particles to first number with integer sq. root

    initialize_xv_arrays()
    
    xr = R * ( np.sqrt(np.pi/(4 * nu)) - 1 )
    LX = 2 * np.sqrt(npart * np.pi/(4* nu) )  
    LY = LX # sistema cuadrado si se determina tamano a partir de nu

    print('xr: ', xr)
    print('LX: ', LX)
    print('LY: ', LY)
    
    # colocar partículas en una red cuadrada
    # para una fraccion de empaquetamiento dada
    # evintando solapamiento
    ii = 0 # particle pseudo-index for 
    for i in range(1,NL+1):
        for j in range(1,NL+1):
            x[ii] = (R + xr) + (i-1) * 2 * (R+ xr)
            y[ii] = (R + xr) + (j-1) * 2 * (R+ xr)
            ii = ii + 1

    for i in range(npart):
        x[i] = x[i] - LX * 0.5
        y[i] = y[i] - LY * 0.5


def initialize_vels_normal():

    #   velocidades aleatorias para las velocidades, distribucion gaussiana
    for i in range(npart):
        vx[i] = np.random.randn()
        vy[i] = np.random.randn()


def propaga(dt):
    #   avanza las particulas con v cte un intervalo de tiempo dt
    global vx, vy, x, y
    x = x + vx * dt
    y = y + vy * dt
        

def midedist(i,j):
    global x, y
    dx = x[i] - x[j]
    dy = y[i] - y[j]
    dist2 = (dx*dx + dy*dy)- 4*R*R


def tcol(i,j):
    #   calcula los tiempos de colision p-p. para un par (i,j)
    global vx, vy, x, y
#    print(LX)
    dx = x[i] - x[j]
    dy = y[i] - y[j]
    dvx = vx[i] - vx[j]
    dvy = vy[i] - vy[j]
    drdv = dx*dvx + dy*dvy
    # estructura condicional de colision p-p
    # condicion de acercamiento
    if drdv > 0:
        vct = float('inf')
    else:
        dist2 = (dx*dx + dy*dy) - 4*R*R # distancia instantanea entre dos particulas
        raiz=drdv*drdv - dist2 * (dvx*dvx + dvy*dvy) # condicion de solucion real en la condicion de col.
        if raiz < 0:
            vct = float('inf')
            # si hay sol. real, guarda en dt el tiempo de col.
        else:
            vdt = dist2/(math.sqrt(raiz)-drdv)
            # posicion de la colision. si en realidad la colision ocurriria fuera del sistema, descartala
            xicol = x[i] + vx[i]*vdt
            yicol = y[i] + vy[i]*vdt
            xjcol = x[j] + vx[j]*vdt
            yjcol = y[j] + vy[j]*vdt
            # estructura condicional de col. fuera del sistema
            if math.fabs(xicol) > LXR:
                vdt = float('inf')
            elif math.fabs(xjcol) > LXR:
                vdt = float('inf')
            elif math.fabs(yicol) > LYR:
                vdt = float('inf')
            elif math.fabs(yjcol) > LYR:
                vdt = float('inf')
            else:
                # coloca en la lista de colisiones ordenada de menor a mayor
                # usa un algoritmo rapido 'binary search' para la colocacion
                bisect.insort(listacol,[vdt,[i,j]])


def tpcol(i):
    #   calcula los tiempos de colision particula-muro. para una particula i.
    #   identificadores de pared: -1 (izq.), -2 (inf.), -3 (dcha.), -4 (sup.)
    global vx, vy, x, y

    if vx[i] == 0:
        tx = float('inf')
    elif vx[i] < 0:
        ltx = [-(LXR+x[i])/vx[i],-1]
    elif vx[i] > 0:
        ltx = [(LXR-x[i])/vx[i],-3]
        
    if vy[i] == 0:
        ty = float('inf')
    elif vy[i] < 0:
        lty = [-(LYR+y[i])/vy[i],-2]
    elif vy[i] > 0:
        lty = [(LYR-y[i])/vy[i],-4]

    ltm = sorted( [ltx,lty], key=itemgetter(0) )
    vdt = ltm[0][0]
    im = ltm[0][1]
    bisect.insort( listacol, [vdt,[i,im]] )


def pcolisiona(ii):
    # actualiza velocidad de part. que colisiona con pared
    global vx, vy, x, y   
    if ii[1]==-1 or ii[1]==-3:
        vx[ii[0]] = -vx[ii[0]]
    elif ii[1]==-2 or ii[1]==-4:
        vy[ii[0]] = -vy[ii[0]]


def colisiona(par):
    # actualiza velocidades de parts. que han colisionado
    global vx, vy, x, y
    # la 1a particula la llamamos i y la 2a, j
    i=par[0]
    j=par[1]
    
    dx=x[i]-x[j]
    dy=y[i]-y[j]
    
    # construye sigma_ij unitario
    sigma_norma=math.sqrt(dx*dx+dy*dy)
    sigmax=dx/sigma_norma
    sigmay=dy/sigma_norma
    
    # construye g \cdot sigma (g, vel relativa)
    gsigma=(vx[i]-vx[j])*sigmax+(vy[i]-vy[j])*sigmay
    
    # actualiza vel. de 1a. part.
    vx[i]=vx[i]-0.5*(1+alfa)*gsigma*sigmax
    vy[i]=vy[i]-0.5*(1+alfa)*gsigma*sigmay
    
    # actualiza vel. de 2a. part.
    vx[j]=vx[j]+0.5*(1+alfa)*gsigma*sigmax
    vy[j]=vy[j]+0.5*(1+alfa)*gsigma*sigmay


def write_micr_state(ja):
    global vx, vy, x, y
    # formatea el nombre de archivo de posiciones y escribelo en disco

    # print ("####### it: ########", it) # imprime it (n. de cols.) #opcional
    # print ("####### no. archivo: ########", ja) # n. de archivo #opcional
    inum='{0:04d}'.format(ja)

    # nombre='/xy'+inum+'.dat'
    nombre='C:/Users/1812p/Desktop/UNIVERSIDAD/MUSCI/2º CUATRIMESTRE/Física Estadística Computacional/PRACTICAS/Fran Vega/pyMD-master/datos/xy'+inum+'.dat'
    aa = pd.DataFrame( np.array([[x[i], y[i]] for i in range(npart)]))
    aa.to_csv(nombre, sep = '\t', \
              header = ['x', 'y'], index = False, float_format ='%10.2f')
    # with open(nombre,'w') as archivo:
    #     archivo.write('x\ty\n')
    #     for i in range(npart):
    #         archivo.write('{0:10.2f} {1:10.2f}\n'.format(x[i],y[i]))
    # archivo.closed

    # formatea el nombre de archivo de posiciones 
    nombre='C:/Users/1812p/Desktop/UNIVERSIDAD/MUSCI/2º CUATRIMESTRE/Física Estadística Computacional/PRACTICAS/Fran Vega/pyMD-master/datos/vxvy'+inum+'.dat'
    bb = pd.DataFrame( np.array([[vx[i], vy[i]] for i in range(npart)]))
    bb.to_csv(nombre, sep = '\t', \
              header = ['vx', 'vy'], index = False, float_format ='%10.2f')
    # with open(nombre,'w') as archivo:
    #     archivo.write('vx vy\n')
    #     for i in range(npart):
    #         archivo.write('{0:10.2f} {1:10.2f}\n'.format( vx[i], vy[i]))
    # archivo.closed    
        

def calculate_averages(ja):
    # measures average fields
    global temp, a2, vx, vy, x, y
    temp[ja]=0.
    a2[ja]=0.
    for i in range(npart):
        vv=vx[i]*vx[i]+vy[i]*vy[i]
        temp[ja]=temp[ja]+vv
        a2[ja]=a2[ja]+vv*vv
    temp[ja]=temp[ja]/npart
    a2[ja]=a2[ja]/(temp[ja]*temp[ja]*npart)
    a2[ja]=(a2[ja]-2.0)*0.5


def write_averages_evol():
    # wites average fields evolution, in a final file
    nombre='C:/Users/1812p/Desktop/UNIVERSIDAD/MUSCI/2º CUATRIMESTRE/Física Estadística Computacional/PRACTICAS/Fran Vega/pyMD-master/datos/temp.dat'
    xy= pd.DataFrame( np.array([[temp[i],a2[i]] for i in range(len(temp))]) )
    xy.to_csv(nombre, sep='\t',\
               header =['T','a2'] , index=False,float_format='%8.5f')


def generate_gif(name):
    # genera un gif con la simulacion
    
    global R, npart, nt, LX, LY
    
    def create_frame(t):
        # pinta fotogramas del gif 
        fig, axs = plt.subplots(figsize = [6,6])
        for i in data.index:
            circle = plt.Circle((x[i], y[i]), R, color = 'b')
            axs.add_artist(circle)
        plt.xlim([-LX*0.5, LX*0.5])
        plt.ylim([-LY*0.5, LY*0.5])
        plt.savefig(f'C:/Users/1812p/Desktop/UNIVERSIDAD/MUSCI/2º CUATRIMESTRE/Física Estadística Computacional/PRACTICAS/Fran Vega/pyMD-master/img/img_{t}.png', 
                    transparent = False,  
                    facecolor = 'white')
        plt.close()
    
    time = list(range (0, nt+1) )
    t = 0
    dir_path = 'C:/Users/1812p/Desktop/UNIVERSIDAD/MUSCI/2º CUATRIMESTRE/Física Estadística Computacional/PRACTICAS/Fran Vega/pyMD-master/datos/'
    for file in os.listdir(dir_path):
        if 'xy' in file:
            full_name = os.path.abspath(os.path.join(dir_path, file))
            data = pd.read_csv(full_name, delimiter = '\t')
            x = data['x']; y = data['y']
            create_frame(t)
            t += 1
            
        
    frames = []
    for t in time:
        image = imageio.v2.imread(f'C:/Users/1812p/Desktop/UNIVERSIDAD/MUSCI/2º CUATRIMESTRE/Física Estadística Computacional/PRACTICAS/Fran Vega/pyMD-master/img/img_{t}.png')
        frames.append(image)
        
    # Combinamos todos los frames en un GIF
    imageio.mimsave('C:/Users/1812p/Desktop/UNIVERSIDAD/MUSCI/2º CUATRIMESTRE/Física Estadística Computacional/PRACTICAS/Fran Vega/pyMD-master/'+name+'.gif',    #output gif
                    frames,             #array of input frames
                    fps = 5,
                    loop = 1)            #opt: frames per second 


    
# %% MAIN

#### INICIALIZACION. coloca particulas, asigna vels. iniciales y calcula t. de cols. iniciales
#   genera posiciones aleatorias -no solapantes- para las particulas
#   este algoritmo de colocacion es necesario sustituirlo para densidades altas por otro mejor

bar = Bar('Archivos', max = nt+1)
square = False

if square==True:
    print("Estado inicial en red cuadrada\n")
    # fraccion de empaquetamiento
    nu = 0.4
    print("Fraccion de empaquetamiento: ", nu, "\n")
    initialize_square()
    # medio tamano del sistema menos medio radio (para situar las particulas)
    # con origen de coordenadas en el centro del sistema
    LXR = LX * 0.5 - R
    LYR = LY * 0.5 - R
else:
    # tamano del sistema
    LX = 16*R
    LY = 16*R
    # medio tamano del sistema menos medio radio (para situar las particulas)
    # con origen de coordenadas en el centro del sistema
    LXR = LX * 0.5 - R
    LYR = LY * 0.5 - R

    # inicializa
    initialize_random()

    
initialize_vels_normal()
write_micr_state(0)
bar.next()


#### bucle en particulas. Calcula tiempos iniciales de colision 

for i in range(npart-1):
    for j in range(i+1,npart):
        tcol(i,j)   # para todos los pares de particulas (i,j) con j>i
for i in range(npart):
    tpcol(i)    # con la pared

it=0


######  inicia bucle temporal principal  (en cols.)  #######

for it in range(1,nt+1):

    # el tiempo mas corto es la col. que de verdad ocurre
    # guarda como tiempo de col. real (1a. componente de 1er elemento de listacol)
    dt=listacol[0][0]*(1-tol)

    # guardamos las etiquetas del par de particulas que colisionan
    # ( los dos elementos de la 2a. componente del primer elemento de listacol)
    ij=listacol[0][1]
    
    # filtra lista de colisiones, eliminando las cols. que ya no ocurriran
    # es decir, se borran los t. de col. calculados para las particulas que colisionaron
    # y por tanto cambiaron su trayectoria
    
    # elimina antiguas colisiones de 1a particula con pares de mayor indice
    listacol=list(filter(lambda x: x[1][0]!=ij[0] , listacol))
    # elimina antiguas colisiones de 1a particula con pares de menor indice
    listacol=list(filter(lambda x: x[1][1]!=ij[0] , listacol))
    
    if ij[1]>0: # si la segunda particula no es un muro:
        # elimina antiguas colisiones de 2a particula con pares de mayor indice
        listalcol=list(filter(lambda x: x[1][0]!=ij[1] , listacol))
        # elimina antiguas colisiones de 2a particula con pares de menor indice
        listacol=list(filter(lambda x: x[1][1]!=ij[1] , listacol))

    
    t=t+dt # actualiza el tiempo fisico (en escala reducida del sistema)

    # y avanza los tiempos de colision, actualizandolos (porque el t ha avanzado dt)
    limit=range(len(listacol))
    c=[[listacol[i][0]-dt,listacol[i][1]] for i in limit]
    listacol=c
    

    # actualiza primero las posiciones de las parts., 
    # justo hasta la colision que primero ocurre
    propaga(dt)


    # actualiza vels. de part(s). que ha(n) colisionado si la col. es con un muro (pcolisiona)
    # la condicion de colision con un muro es que la "segunda particula" tiene indice negativo
    if ij[1]<0:
        pcolisiona(ij)
    # en caso contrario, la col. es entre dos part. (colisiona)
    else:
        colisiona(ij)
    
    # ahora calculamos los tiempos de col. nuevos para las nuevas trayectorias 
    # de las particulas que colisionarion, 
    # las funciones tcol y tpcol ademas recolocaran ordenadamente esos t en listacol
   
    # primera particula
    i=ij[0]
    # nuevos tiempos de col. de la 1a particula que acaba de colisionar
    tpcol(i)    # con la pared
    
    for j in range(i):
        tcol(j,i)   # para todos los pares de particulas (i,j) con j<i
        
    for j in range(i+1,npart):
        tcol(i,j)   # para todos los pares de particulas (i,j) con j>i
    
    # segunda particula, solo si no es un muro (y por tanto, tiene indice positivo)
    if ij[1]>0:
        i=ij[1]
        # nuevos tiempos de col. de la 2a particula que acaba de colisionar
        tpcol(i)    # con la pared
        
        for j in range(i):
            tcol(j,i)   # para todos los pares de particulas (i,j) con j<i
        
        for j in range(i+1,npart):
            tcol(i,j)   # para todos los pares de particulas (i,j) con j>i
    

##  Escribe pos. y vels. iterativamente

    if (it%utermo==0):
        ia=int(it/utermo)
        write_micr_state(ia)
        bar.next()
        # medicion de T y a2 
        calculate_averages(ia)

write_averages_evol()

name_gif = 'Part_25_col_10'
generate_gif(name_gif)
    
######  FIN bucle temporal principal  (en cols.)  #######
