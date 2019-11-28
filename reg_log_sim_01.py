# Regresion logistica simple

import matplotlib.pyplot as plt
from scipy.special import expit 
import numpy as np 
from scipy.optimize import fmin
from scipy.optimize import fmin_bfgs

def sigmoide(z):
    # sigmoide, Calcula la funcion sigmoidea
    # J = sigmoide(z) calcula la sigmoide de z.
    g = np.zeros(z.shape)

    # g = 1/(1 + np.exp(-z))
    g = expit(z)

    return g

def funcionCosto(theta, X, y, retornar_gradiente = False):
    # funcionCosto, calcula el costo y el gradiente para la regresion logistica
    # J = funcionCosto(theta, X, y) calcula el costo utilizando theta como
    # parametro para la regresion logistica y el gradiente de el costo
    # w.r.t.  para los parametros.

    m = len(y) # numero de ejemplos de entrenamiento
    J = 0
    grad = np.zeros(theta.shape)

    # dadas las siguientes dimensiones:
    # theta.shape = (n + 1, 1)
    # X.shape     = (m, n + 1)
    # La ecuacion theta considerando X veces es
    # np.dot(X, theta)
    # para obtener un vector (m,1)
    # dado
    # y.shape = (m,)
    # se traspone a (m,1)
    # np.log(sigmoidea(np.dot(X, theta))), tanto como
    # np.log( 1 - sigmoid( np.dot(X,theta)))
    # para obtener (1, m) vectores para ser mutuamente añadido,
    # y cuyos elementos se suman para formar un escalar.
    one = y * np.transpose(np.log(sigmoide(np.dot(X, theta))))
    two = (1 - y) * np.transpose(np.log(1 - sigmoide(np.dot(X, theta))))
    J = -(1./ m) * (one + two).sum()

    # se necesita n + 1 gradientes. 
    # nota que: 
    # y.shape = (m,)
    # sigmoide(np.dot(X,theta)).shape = (m, 1)
    # entonces transponemos lo último, restamos y, obtenemos un vector de (1, m)
    # multiplicando dicho vector por X, con dimension 
    # X.shape = (m, n+1), 
    # y obtener un vector (1, n + 1), que se traspone
    # esta última multiplicación vectorizada se encarga de la suma.
    grad = (1./m) * np.dot(sigmoide(np.dot(X, theta) ).T - y, X).T

    if retornar_gradiente == True:
        return J, np.transpose(grad)
    elif retornar_gradiente == False:
        return J # para utilizar en las funciones de optimizacion fmin/fmin_bfgs

def graficarDatos(X, y):
    # graficarDatos, grafica los datos como puntos X, y en una nueva figura 
    # graficarDatos(X, y) grafica los puntos de los datos representando con + los ejemplos positivos
    # y con o los ejemplos negativos. Se asume que X es una matriz Mx2

    # Encontrar indices de los ejemplos positivos y negativos
    pos = np.where(y == 1)
    neg = np.where(y == 0)

    # graficar! indexando [0] al final si es necesario para la creacion de una adecuada leyenda
    p1 = plt.plot(X[pos, 0], X[pos, 1], marker='+', markersize=9, color='k')[0]
    p2 = plt.plot(X[neg, 0], X[neg, 1], marker='o', markersize=7, color='y')[0]
    
    return plt, p1, p2

def graficarBordesDecision(theta, X, y):
    # graficarBordesDecision Grafica puntos X y y de los datos en una nueva figura
    # con los bordes de decision definidos por theta
    # graficarBordesDecision(theta, X, y) grafica los datos como puntos con + los positivos
    # y con o los ejemplos negativos. Se supone que X una:
    # 1) Matriz Mx3, donde la primera columna es una columna de unos para la intercepcion
    # 2) matriz MxN, N>3, donde la primera columna es all-ones

    # Graficar Data
    #fig = plt.figure()

    plt, p1, p2 = graficarDatos(X[:, 1:3], y)
    
    if X.shape[1] <= 3:
        # Solamente se necesitan 2 puntos para definir una linea, así que se elige dos puntos finales
        plot_x = np.array([min(X[:, 1]) - 2,  max(X[:, 1]) + 2])

        # Calcular la linea de calcular la línea de límite de decisión
        plot_y = (-1./theta[2]) * (theta[1] * plot_x + theta[0])

        # Graficar, y ajustar los ejes para mejora la visualizacion
        p3 = plt.plot(plot_x, plot_y)
        
        # Leyenda, específica para el ejercicio.
        plt.legend((p1, p2, p3[0]), ('Admitido', 'No admitido', 'Límite de decisión'), numpoints = 1, handlelength = 0.5)

        plt.axis([30, 100, 30, 100])

        plt.show(block=False)
    else:
        # Rango de la cuadrícula
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))
        # Evaluar z = theta * x sobre la cuadricula
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = np.dot(mf.mapFeature(np.array([u[i]]), np.array([v[j]])), theta)
        z = np.transpose(z) # Es importante trasponer z antes de llamar al limite

        # Graficar z = 0
        # Tenga en cuenta que necesita especificar el nivel 0
        # Obtenemos colecciones [0] para que podamos mostrar una leyenda correctamente
        p3 = plt.contour(u, v, z, levels = [0], linewidth = 2).collections[0]
        
        # Leyenda, especifica para el ejercicio
        plt.legend((p1, p2, p3),('y = 1', 'y = 0', 'Limite de decision'), numpoints = 1, handlelength = 0)
        plt.show(block=False)

def predecir(theta, X):
    # predecir, Predice si la etiqueta es 0 o 1 usando los parámetros de regresión logística 
    # aprendidos utilizando theta
    # p = predecir(theta, X) calcula las predicciones para X utilizando un 
    # umbral a 0.5 (i.e., si sigmoidea(theta'*x) >= 0.5, predice 1)

    m = X.shape[0] # Numero de ejemplos de entrenamiento
    p = np.zeros((m, 1))

    sigValue = sigmoide(np.dot(X, theta))
    p = sigValue >= 0.5

    return p

# Cargar datos
# La primera de dos columnas contiene el resultado de un examen y la tercera columna
# contiene la etiqueta.

data = np.loadtxt('ex2data1.txt', delimiter = ",")
X = data[:, :2]
y = data[:, 2]

# ==================== Parte 1: Graficar ====================
# Se inicia el ejercicio graficando los datos para entender el problema con el que estamos trabajando 

print('Graficando los datos ejemplo identificando (y = 1) con + y (y = 0) con o.')

plt, p1, p2 = graficarDatos(X, y)

# Etiquetas y leyendas
plt.xlabel('Resultados examen 1')
plt.ylabel('Resultados examen 2')
plt.legend((p1, p2), ('Admitido', 'No admitido'), numpoints = 1, handlelength = 0)

plt.show(block=False) # evita que se tenga que cerrar el grafico para continuar con la ejecucion del programa

input('Programa detenido. Precione <Enter> para continuar.\n')
plt.close()

# ============ Part 2: Calcular el Costo y el Gradiente ============
# Configurar adecuadamente la matriz, y agregar unos para el termino de intececcion
m, n = X.shape
X_padded = np.column_stack((np.ones((m, 1)), X)) 

# Inicializar parámetros de ajuste
initial_theta = np.zeros((n + 1, 1))

# Calcular y mostrar el costo inicial y el gradiente
costo, gradiente = funcionCosto(initial_theta, X_padded, y, retornar_gradiente = True)

print('Costo con theta inicial (ceros): {:f}'.format(costo))
print('Gradiente con theta inicial (ceros):')
print(gradiente)

input('Programa detenido. Precione <Enter> para continuar.\n')

# ============= Part 3: Optimizacion utilizando fmin (y fmin_bfgs)  =============
# Se utilizara la funcion incorporada (fmin) para encontrar el parametro optimo de theta
# Ejecuta fmin y fmin_bfgs para obtener el theta optimo
# Esta funcion retorna theta y el costo

myargs = (X_padded, y)
theta = fmin(funcionCosto, x0 = initial_theta, args = myargs)
theta, cost_at_theta, _, _, _, _, _ = fmin_bfgs(funcionCosto, x0 = theta, args = myargs, full_output = True)

# Imprimir theta en la pantalla
print('Costo a partir de theta encontrado por fmin: {:f}'.format(cost_at_theta))
print('theta:'),
print(theta)

# Gaficar contornos
graficarBordesDecision(theta, X_padded, y)
plt.show(block=False) 

input('Programa detenido. Precione <Enter> para continuar.\n')


# ============== Parte 4: Predecir y Precision ==============
# Se utilizara un modelo de regresion logistica para 
# predecir la probabilidad de un estudiante con una calificacion de 45 en su examen 1
# y 85 en su examen 2 pueda ser admitido
# Además, calculara la precisión y el conjunto de pruebas de nuestro modelo.

prob = sigmoide(np.dot(np.array([1, 45, 85]), theta))
print('Para un estudiante con calificaciones de 45 y 85, se predice una probabilidad de admision de {:f}'.format(prob))

# Calcular la precision sobre nuestro conjunto de entrenamiento
p = predecir(theta, X_padded)

print('Precision en el entrenamiento: {:f}'.format(np.mean(p == y) * 100))

input('Programa detenido. Precione <Enter> para continuar.\n')
