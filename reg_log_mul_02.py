# Regresion logistica con multiples variables y regularizacion

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_bfgs
from scipy.special import expit 

def sigmoide(z):
    # SIGMOIDE funcion para calcular la sigmoide
    # J = SIGMOIDE(z) calcula la sigmoide de z.

    g = np.zeros(z.shape)
    # g = 1/(1 + np.exp(-z))
    g = expit(z)

    return g

def funcionCostoRegularizada(theta, X, y, lambda_reg, return_grad = False):
    # FUNCIONCOSTOREGULARIZADA Calcula el costo y el gradiente para la regresion logistica con regularización
    # J = FUNCIONCOSTOREGULARIZADA(theta, X, y, lambda) calcula el costo de usar
    # theta como parámetro para la regresión logística regularizada 
    # y el gradiente del costo w.r.t. a los parametros.

    # Inicializa algunos valores utiles
    m = len(y) # numero de ejemplos para entrenamiento

    J = 0
    grad = np.zeros(theta.shape)
    
    # Calcular el costo de un valor theta particular elegido.
    # Se debe establecer a J el costo.
    # Calcula la derivada parcial y establece el grado de las derivadas parciales
    # de el costo w.r.t. en cada parametro en theta, tomando principalmente de funcionCosto
    # y agregando el termino de regularizacion
    # tenga en cuenta que no solo tomamos todo el theta,
    # sino que solo n del tamaño de los elementos n + 1 (theta) es igual a [n + 1 1], 
    # así que tomamos el primer elemento de eso (en tamaño (theta, 1)) 
    # para la expresión theta (2: tamaño (theta, 1))

    one = y * np.transpose(np.log(sigmoide(np.dot(X, theta))))
    two = (1 - y) * np.transpose(np.log(1 - sigmoide(np.dot(X, theta))))
    reg = (float(lambda_reg) / (2 * m)) * np.power(theta[1:theta.shape[0]], 2).sum()
    J = -(1./m) * (one + two).sum() + reg

    # se aplica a j = 1, 2, ..., n - NOT to j = 0
    grad = (1./m) * np.dot(sigmoide(np.dot(X, theta) ).T - y, X).T + ( float(lambda_reg) / m ) * theta

    # the case of j = 0 (recall that grad is a n+1 vector)
    # since we already have the whole vectorized version, we use that
    grad_no_regularization = (1./m) * np.dot(sigmoide( np.dot(X,theta) ).T - y, X).T

    # and then assign only the first element of grad_no_regularization to grad
    grad[0] = grad_no_regularization[0]

    if return_grad == True:
        return J, grad.flatten()
    elif return_grad == False:
        return J 

def graficarDatos(X, y):
    # GRAFICARDATOS traza los datos de los puntos X e y en una nueva figura 
    # graficarDatos(X, y) traza los puntos de datos con + para los ejemplos positivos
    # y con o para los ejemplos negativos. X es asumida como una matriz Mx2.

    # Encontrar los indices de los ejemplos positivos y negativos
    pos = np.where(y==1)
    neg = np.where(y==0)

    # trazar! [0] La indexación al final es necesaria para la creación correcta de la leyenda
    p1 = plt.plot(X[pos, 0], X[pos, 1], marker = '+', markersize = 9, color = 'k')[0]
    p2 = plt.plot(X[neg, 0], X[neg, 1], marker = 'o', markersize = 7, color = 'y')[0]
    
    return plt, p1, p2


def mapearCaracteristica(X1, X2):
    # MAPEARCARACTERISTICA Función de mapeo de características a características polinomiales.
    # MAPEARCARACTERISTICA(X1, X2) mapea las dos caracteristicas de entrada
    # a caracteristicas cuadraticas usadas en el ejercicio de regularizacion.
    # Devuelve una nueva matriz de funciones con más funciones, que comprende 
    # X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    # para un total de 1 + 2 + ... + (gado + 1) = ((grado + 1) * (grado + 2)) / 2 columnas
    # Las entradas X1, X2 deben ser del mismo tamaño
    
    degree = 6
    out = np.ones((X1.shape[0], sum(range(degree + 2)))) # También se puede usar ((grado + 1) * (grado + 2)) / 2 en lugar de la suma
    curr_column = 1
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out[:, curr_column] = np.power(X1, i - j) * np.power(X2, j)
            curr_column += 1

    return out

def predecir(theta, X):
    # PREDICIR, Predecir si la etiqueta es 0 o 1 usando la logística aprendida
    # parametros theta de la regresion 
    # p = PREDECIR(theta, X) calcula la prediccion para un X utilizando un umbral 
    # a 0.5 (i.e., si sigmoidea(theta'*x) >= 0.5, predict 1)

    m = X.shape[0] # Numero de ejemplos de entrenamiento

    p = np.zeros((m, 1))

    sigValue = sigmoide(np.dot(X, theta))
    p = sigValue >= 0.5

    return p

def trazarLimitesDecision(theta, X, y):
    # TRAZARLIMITESDECISION, Traza los puntos de datos X e Y en una nueva figura con el límite de decisión definido por theta
    # TRAZARLIMITESDECISION(theta, X, y) traza los puntos de datos con + para los ejemplos positivos y o para los ejemplos negativos. 
    # Se supone que X es una
    # 1) Mx3 matrix, where the first column is an all-ones column for the 
    # intercept.
    # 2) MxN, N>3 matrix, where the first column is all-ones
    
    # Trazar datos
    # fig = plt.figure()

    plt, p1, p2 = graficarDatos(X[:, 1:3], y)
    
    if X.shape[1] <= 3:
        # Solo se requieren 2 puntos para definir una linea, se elijen los puntos finales
        plot_x = np.array([min(X[:, 1]) - 2,  max(X[:, 1]) + 2])

        # Calcula el limite de descision
        plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0])

        # Trazar y ajustar los ejes para mejorar la vista
        p3 = plt.plot(plot_x, plot_y)
        
        # Leyenda, specifica para el ejercicio
        plt.legend((p1, p2, p3[0]), ('Admitido', 'No admitido', 'Limite de decision'), numpoints=1, handlelength=0.5)

        plt.axis([30, 100, 30, 100])

        plt.show(block=False)
    else:
        # Rango de la grilla
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros(( len(u), len(v) ))
        # Evalua z = theta * x sobre la grilla
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = np.dot(mapearCaracteristica(np.array([u[i]]), np.array([v[j]])),theta)
        z = np.transpose(z) # importante trasponer z antes de llamar a contornear

        # Trazar z = 0
        # Se necesita especificar el nivel 0
        # obtenemos coleccions[0] para mostrar una leyenda correctamente
        p3 = plt.contour(u, v, z, levels = [0], linewidth = 2).collections[0]
        
        # Leyenda, especifica para el ejercicio
        plt.legend((p1,p2, p3),('y = 1', 'y = 0', 'Limites de descision'), numpoints=1, handlelength=0)

        plt.show(block=False)

# Cargar datos
# Las primeras dos columnas contienen los resultados de las pruebas y 
# la tercera columna contiene la etiqueta.

data = np.loadtxt('ex2data2.txt', delimiter=",")
X = data[:, :2]
y = data[:, 2]

plt, p1, p2 = graficarDatos(X, y)

# Etiquetas y leyendas
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend((p1, p2), ('y = 1', 'y = 0'), numpoints = 1, handlelength = 0)

plt.show(block=False) # evita tener que cerrar la gráfica para avanzar con el programa

input('Programa en pausa. Precione <Enter> para continuar.\n')


# Parte 1: Regresion logistica regularizada ============
# En esta parte, se le proporciona un conjunto de datos con puntos de datos 
# que no se pueden separar linealmente. Sin embargo, aún desea utilizar 
# la regresión logística para clasificar los puntos de datos.
# Para hacerlo, introduce más caracteristicas para usar, en particular, 
# agrega funciones polinomiales a nuestra matriz de datos (similar a la regresión polinomial).

# Agregar caracteristicas polinomiales

# mapFeature agrega una columna de unos para el término de intercepción

X = mapearCaracteristica(X[:, 0], X[:, 1])
m, n = X.shape

# Inicializar parametros de ajuste
initial_theta = np.zeros((n, 1))

# Ajusta el parámetro de regularización lambda a 1
lambda_reg = 0.1

# Calcula y muestra el costo inicial
# el gradiente es muy grande para mostrar en este ejercicio
cost = funcionCostoRegularizada(initial_theta, X, y, lambda_reg)

print('Costo a theta inicial (zeros): {:f}'.format(cost))
# print('Gradiente a theta inicial (zeros):')
# print(grad)

input('Programa en pausa. Precione <Enter> para continuar.\n')


# ============= Part 2: Regularizacion y precision =============

# Inicializar los parametros de ajuste
initial_theta = np.zeros((n, 1))

# Establece el parametro de regularizacion lambda a 1 (se pude ensayar variaciones)
lambda_reg = 1

# Ejecutar fmin_bfgs para obtener el theta optimo
# Esta funcion retorna theta y el costo
myargs=(X, y, lambda_reg)
theta = fmin_bfgs(funcionCostoRegularizada, x0 = initial_theta, args = myargs)

# Trazar limite
trazarLimitesDecision(theta, X, y)

# Etiquetas, titulos y leyendas
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.title('lambda = {:f}'.format(lambda_reg))

# % Calcular la precision en el conjunto de entrenamiento
p = predecir(theta, X)

print('Precision en entrenamiento: {:f}'.format(np.mean(p == y) * 100))

input('Programa en pausa. Prescione <Entre> para continuar.\n')

