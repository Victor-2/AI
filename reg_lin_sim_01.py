# Regresión lineal
# x hace referencia al tamaño de poblacion en 10,000 miles
# y hace referencia al ingreso en $10,000 miles

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def calcularCosto(X, y, theta):
    J = 0
    s = np.power((X.dot(theta) - np.transpose([y])), 2)
    J = (1.0 / (2 * m)) * s.sum(axis = 0)

    return J

def descensoGradiente(X, y, theta, alpha, num_iteraciones):
    m = len(y) # Numero de ejemplos para entrenamiento
    J_history = np.zeros((num_iteraciones, 1))

    for i in range(num_iteraciones):
        theta = theta - alpha * (1.0 / m) * np.transpose(X).dot(X.dot(theta) - np.transpose([y]))
        J_history[i] = calcularCosto(X, y, theta)
        # print(J_history[i])

    return theta

def graficarDatos(x, y):
    plt.plot(x, y, 'rx', markersize = 10, label = 'Datos de entrenamiento')
    plt.xlabel('Poblacion de la ciudad en 10,000 miles')
    plt.ylabel('Beneficio en $10,000 miles')
    plt.show(block = False) # evita tener que cerrar la gráfica para avanzar con la ejecucion del programa

# ======================= Cargar datos =======================
print('Cargando datos...')

data = np.loadtxt('ex1data1.txt', delimiter = ",")
X = data[:,0]
y = data[:,1]
m = len(y) # numero de ejemplos de entrenamiento

# ======================= Graficar =======================
print('Graficando datos...')

graficarDatos(X, y)
input('Programa en pausa. Presione <enter> para continuar.\n')

# =================== Descenso por el gradiente ===================
print('Ejecutando descenso por el gradiente...')

X_x0agregado = np.column_stack((np.ones((m, 1)), X)) # Agrega una columna de unos a X
theta = np.zeros([2, 1]) # inicializa los valores de los parametros de theta

# Establecer valores para algunos parametros utilizados en el descenso por el gradiente
iteraciones = 1500
alpha = 0.01

# Calcular y mostrar el costo inicial
print(calcularCosto(X_x0agregado, y, theta))

# Ejecutar el descenso por el gradiente
theta = descensoGradiente(X_x0agregado, y, theta, alpha, iteraciones)

# Imprimir en pantalla los valores de theta
print('Theta encontrado por el descenso del gradiente: ')
print("{:f}, {:f}".format(theta[0,0], theta[1,0]))

# Trazar el ajuste lineal
plt.plot(X, X_x0agregado.dot(theta),'-', label='Regresión lineal')
plt.legend(loc = 'lower right')
plt.draw()

# Predecir valores para los tamaños de poblacioni de 35,000 y 70,000
prediccion1 = np.array([1, 3.5]).dot(theta)
print("Para una poblacion = 35,000, se predice un beneficio de {:f}".format( float(prediccion1 * 10000)))
prediccion2 = np.array([1, 7]).dot(theta)
print('Para una poblacion = 70,000, se predice un beneficio de {:f}'.format( float(prediccion2 * 10000)))
input('Programa en pausa. Presione <enter> para continuar.\n')

## ============= Visualizacion J(theta_0, theta_1) =============
print('Visualizando J(theta_0, theta_1)...')

# arreglos de theta sobre la que calcularemos J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# Inicializa J_vals como una matriz de ceros
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Rellena J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = [[theta0_vals[i]], [theta1_vals[j]]]
        J_vals[i,j] = calcularCosto(X_x0agregado, y, t)

# Debido a la forma en que funcionan las rejillas de malla en el comando de navegación, 
# necesitamos transponer J_vals antes de visualizar, o de lo contrario los ejes se voltearán

J_vals = np.transpose(J_vals)

# Graficar la superficie

fig = plt.figure()
ax = fig.gca(projection = '3d')
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals) # necesario para gaficos 3D
surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap = cm.coolwarm, rstride=2, cstride=2)
fig.colorbar(surf)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.show(block=False)

# Gaficar contorno
fig = plt.figure()
ax = fig.add_subplot(111) # "111" significa -> cuadrícula 1 × 1, primer subparcela
# Se grafica J_vals como 20 contornos espaciados logaritmicamente entre 0.01 y 100
cset = plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20), cmap=cm.coolwarm)
fig.colorbar(cset)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.plot(theta[0, 0], theta[1, 0], 'rx', markersize = 10, linewidth = 2)
plt.show(block = False)

input('Programa en pausa. Precione enter para terminar la ejecucion del programa.\n')
