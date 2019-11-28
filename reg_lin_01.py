import numpy as np 
import matplotlib.pyplot as plt
import math

def calcularCosto(X, y, theta):
    J = 0
    s = 0
    m = len(y)
    #s = np.power((X.dot(theta) - np.transpose([y])), 2)
    #J = (1.0 / (2 * m)) * s.sum(axis = 0)
    for i in range(m):
        s = s + np.power(((theta[0] + theta[1] * X[i]) - y[i]), 2)
    J = s / 2 * m
    return J

def descensoGradiente(X, y, theta, alpha, num_iteraciones):
    s0 = 0
    s1 = 0
    m = len(y) # Numero de ejemplos para entrenamiento
    #J_history = np.zeros((num_iteraciones, 1))
    J_history = []
    costoMinimo = math.inf
    costoActual = 0
    thetaOptima = [0, 0]
    for i in range(num_iteraciones):
        for j in range(m):
            s0 = s0 + ((theta[0] + theta[1] * X[j]) - y[j]) 
            s1 = s1 + ((theta[0] + theta[1] * X[j]) - y[j]) * X[j]

        theta[0] = theta[0] - alpha * (1.0 / m) * s0 
        theta[1] = theta[1] - alpha * (1.0 / m) * s1
        costoActual = calcularCosto(X, y, theta)
        if abs(costoActual) < costoMinimo:
            costoMinimo = costoActual 
            thetaOptima[0] = theta[0]
            thetaOptima[1] = theta[1]
    return thetaOptima

def graficarDatos(X, y):
    plt.plot(X, y, 'rx', markersize = 10, label = 'Datos de entrenamiento')
    plt.xlabel('Edad estudiantes universitarios')
    plt.ylabel('Ingreso promedio mes')
    plt.show(block = False) # evita tener que cerrar la gráfica para avanzar con la ejecucion del programa

X = [15, 18, 20, 23, 25, 27, 30, 33, 40, 45, 50, 60]
y = [10, 20, 50, 300, 1000, 1200, 5000, 9000, 12000, 20000, 30000, 35000]
m = len(y)

graficarDatos(X, y)

theta = [0, 0]
# Establecer valores para algunos parametros utilizados en el descenso por el gradiente
iteraciones = 15000
alpha = 0.0001

# Calcular y mostrar el costo inicial
print("theta0 = {0}, theta1 = {1} iniciales".format(theta[0], theta[1]))
#print("Costo inicial{0}".format(calcularCosto(X, y, theta)))

# Ejecutar el descenso por el gradiente
theta_optimo = descensoGradiente(X, y, theta, alpha, iteraciones)

# Imprimir en pantalla los valores de theta
print('Theta encontrado por el descenso del gradiente: ')
print("{0}, {1}".format(theta_optimo[0], theta_optimo[1]))

# Trazar el ajuste lineal
y_ajustado = []
for i in range(m):
    y_ajustado.append(theta_optimo[0] + theta_optimo[1] * X[i])

plt.plot(X, y_ajustado,'-', label='Regresión lineal')
plt.legend(loc = 'lower right')
plt.draw()
plt.show()

