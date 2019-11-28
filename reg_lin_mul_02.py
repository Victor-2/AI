# Regresion lineal con multiples variables utilizando la ecuacion de la normal

import numpy as np 
import matplotlib.pyplot as plt

def ecuacionNormal(X, y):
    theta = np.zeros((X.shape[1], 1))
    theta = np.linalg.pinv(np.transpose(X).dot(X)).dot(np.transpose(X).dot(y))

    return theta
# ================ Ecuacion de la normal ================

print('Resolviendo con la ecuacion de la normal...')

## Cargar datos
data = np.loadtxt('ex1data2.txt', delimiter=",")
X = data[:, :2]
y = data[:, 2]
m = len(y) # Numero de ejemplos de entrenamiento

# Añadir el termino de intercepcion de X
X_padded = np.column_stack((np.ones((m,1)), X)) 

# Calcular los parametros a partir de la ecuacion de la normal
theta = ecuacionNormal(X_padded, y)

# Mostrar los resultados obtenidos aplicando la ecuacion de la normal
print('Theta calculado a partir de la ecuacion de la normal:')
print("{:f}, {:f}, {:f}".format(theta[0], theta[1], theta[2]))
print('')


# Estimar el precio de una casa con una superficie de 1650 pies cuadrados, 3 br
house_norm_padded = np.array([1, 1650, 3])
price = np.array(house_norm_padded).dot(theta)


print("Precio pronosticado de una casa de 1650 pies cuadrados y 3 br (usando la ecuacion de la normal:\n ${:,.2f}".format(price))

