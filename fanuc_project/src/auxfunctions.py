import numpy as np
from copy import copy
import rbdl
pi = np.pi

class Robot(object):
    def __init__(self, q0, dq0, ndof, dt):
        self.q = q0    # numpy array (ndof x 1)
        self.dq = dq0  # numpy array (ndof x 1)
        self.M = np.zeros([ndof, ndof])
        self.b = np.zeros(ndof)
        self.dt = dt
        self.robot = rbdl.loadModel('../urdf/fanuc125.urdf')

    def send_command(self, tau):
        rbdl.CompositeRigidBodyAlgorithm(self.robot, self.q, self.M)
        rbdl.NonlinearEffects(self.robot, self.q, self.dq, self.b)
        ddq = np.linalg.inv(self.M).dot(tau-self.b)
        self.q = self.q + self.dt*self.dq
        self.dq = self.dq + self.dt*ddq

    def read_joint_positions(self):
        return self.q

    def read_joint_velocities(self):
        return self.dq



def dh(d, theta, a, alpha):
    """
    Calcular la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.

    """
    sth = np.sin(theta)
    cth = np.cos(theta)
    sa  = np.sin(alpha)
    ca  = np.cos(alpha)
    T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                  [sth,  ca*cth, -sa*cth, a*sth],
                  [0.0,      sa,      ca,     d],
                  [0.0,     0.0,     0.0,   1.0]])
    return T


def fkine_fanuc(q):

    if q[0]<0:
      q[0] = 0
    if q[3]>=0.8:
      q[3]=0.8
    if q[4]>=0.5:
      q[4] = 0.5
    if q[4]<=-0.5:
      q[4] = -0.5
    if q[5]>=0.5:
      q[5] = 0.5
    if q[5]<=-0.5:
      q[5] = -0.5
    # Matrices DH
    T01 = dh(    0,    pi+q[0], -0.320, pi/2)
    T12 = dh(    0,  q[1]+pi/2,  1.075,    0)
    T23 = dh(    0,      -q[2],  0.215,    0)
    T34 = dh( q[3],       pi/2,  1.730,    0)
    T45 = dh(    0,      -q[4],  0.225,    0)
    T56 = dh(    0,       q[5],      0,    0)

    # Efector final con respecto a la base
    T = T01 @ T12 @ T23 @ T34 @ T45 @ T56  
    return T


def jacobian_position(q, delta=0.0001):
    # Alocacion de memoria
    J = np.zeros((3,8))
    # Transformacion homogenea inicial (usando q)
    x = fkine_fanuc(q)[0:3,3] 
    # Iteracion para la derivada de cada columna

    for i in range(8):
        # Copiar la configuracion articular inicial (usar este dq para cada
        # incremento en una articulacion)
        
        dq = copy(q)
        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i] + delta
        # Transformacion homogenea luego del incremento (q+dq)
        dx = fkine_fanuc(dq)[0:3,3] 
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        # J[0:a,i]=(T_inc[]-T[])/delta
        J[:,i] = 1/delta * (dx-x)
    return J

    
def ikine(xdes, q0):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la
    configuracion articular inicial de q0. Emplear el metodo de newton
    """
    fqact = open("/tmp/error_newton.txt", "w")
    epsilon = 0.00001
    max_iter = 10000
    delta = 0.00000001
    q = copy(q0)
    for i in range(max_iter):
      # Main loop
      T = fkine_fanuc(q)
      pos = T[0:3,3]
      er = xdes - pos
      if (np.linalg.norm(er) < epsilon):
        break
      J = jacobian_position(q, delta)
      #Newton
      q = q + np.dot(np.linalg.pinv(J), er)
    return q
   
   
 


def rot2quat(R):
    """
    Convertir una matriz de rotacion en un cuaternion

    Entrada:
      R -- Matriz de rotacion
    Salida:
      Q -- Cuaternion [ew, ex, ey, ez]

    """
    dEpsilon = 1e-6
    quat = 4*[0.,]

    quat[0] = 0.5*np.sqrt(R[0,0]+R[1,1]+R[2,2]+1.0)
    if ( np.fabs(R[0,0]-R[1,1]-R[2,2]+1.0) < dEpsilon ):
        quat[1] = 0.0
    else:
        quat[1] = 0.5*np.sign(R[2,1]-R[1,2])*np.sqrt(R[0,0]-R[1,1]-R[2,2]+1.0)
    if ( np.fabs(R[1,1]-R[2,2]-R[0,0]+1.0) < dEpsilon ):
        quat[2] = 0.0
    else:
        quat[2] = 0.5*np.sign(R[0,2]-R[2,0])*np.sqrt(R[1,1]-R[2,2]-R[0,0]+1.0)
    if ( np.fabs(R[2,2]-R[0,0]-R[1,1]+1.0) < dEpsilon ):
        quat[3] = 0.0
    else:
        quat[3] = 0.5*np.sign(R[1,0]-R[0,1])*np.sqrt(R[2,2]-R[0,0]-R[1,1]+1.0)

    return np.array(quat)


def TF2xyzquat(T):
    """
    Convert a homogeneous transformation matrix into the a vector containing the
    pose of the robot.

    Input:
      T -- A homogeneous transformation
    Output:
      X -- A pose vector in the format [x y z ew ex ey ez], donde la first part
           is Cartesian coordinates and the last part is a quaternion
    """
    quat = rot2quat(T[0:3,0:3])
    res = [T[0,3], T[1,3], T[2,3], quat[0], quat[1], quat[2], quat[3]]
    return np.array(res)


def skew(w):
    R = np.zeros([3,3])
    R[0,1] = -w[2]; R[0,2] = w[1]
    R[1,0] = w[2];  R[1,2] = -w[0]
    R[2,0] = -w[1]; R[2,1] = w[0]
    return R
