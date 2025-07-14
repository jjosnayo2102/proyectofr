import rbdl
import numpy as np

if __name__ == '__main__':

  # Lectura del modelo del robot a partir de URDF (parsing)
  modelo = rbdl.loadModel('../urdf/fanuc125.urdf')
  # Grados de libertad
  ndof = modelo.q_size

  # Configuracion articular inicial (en radianes)
  q = np.array([0.2, 0.4, 0.1, 0.4, 0.2, 0.3, 0, 0])
  # Velocidad inicial
  dq = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0])
  # Aceleracion articular
  ddq = np.array([0.2, 0.5, 0.4, 0.3, 1.0, 0.5, 0, 0])
  
  # Arrays numpy
  zeros = np.zeros(ndof)          # Vector de ceros
  tau   = np.zeros(ndof)          # Para torque
  g     = np.zeros(ndof)          # Para la gravedad
  c     = np.zeros(ndof)          # Para el vector de Coriolis+centrifuga
  mi    = np.zeros(ndof)
  M     = np.zeros([ndof, ndof])  # Para la matriz de inercia
  e     = np.eye(8)               # Vector identidad
  
  # Torque dada la configuracion del robot
  rbdl.InverseDynamics(modelo, q, dq, ddq, tau)
  
  # Parte 1: Calcular vector de gravedad, vector de Coriolis/centrifuga,
  # y matriz M usando solamente InverseDynamics

  # Vector de gravedad
  rbdl.InverseDynamics(modelo, q, zeros, zeros, g)
  g = np.round(g,5)
  print('MATRIZ GRAVEDAD')
  print(np.round(g,3))

  # Vector de Coriolis
  rbdl.InverseDynamics(modelo, q, dq, zeros, c)
  c = c-g
  c = np.round(c,2)
  print('MATRIZ F y C')
  print(c)

  # Matriz M
  for i in range(ndof):
      rbdl.InverseDynamics(modelo, q, zeros, e[i,:], mi)
      M[i,:] = mi - g
    
  print('MATRIZ INERCIA')
  print(np.round(M,4))

  # Parte 3: Verificacion de la expresion de la dinamica
    
  tau2 = M.dot(ddq) + c + g
  print('Vector de torques obtenidos con la funcion InverseDynamics')
  print(np.round(tau2,5))

