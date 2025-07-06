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
        
def trasl(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

def rot_about(axis, angle):
    axis = np.array(axis) / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    R = np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s, 0],
        [y*x*C + z*s, c + y*y*C,     y*z*C - x*s, 0],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C,     0],
        [0,           0,           0,           1]
    ])
    return R

def fkine_fanuc(q):
    q1, q2, q3, q4, q5, q6 = q

    # Joint origins (respecto a base)
    T01 = trasl(0, 0, 0)      @ rot_about([0, 0, 1], q1)
    T12 = trasl(0.050, 0, 0)  @ rot_about([0, 1, 0], q2)
    T23 = trasl(0, 0, 0.330)  @ rot_about([0, -1, 0], q3)
    T34 = trasl(0, 0, 0.035)  @ rot_about([-1, 0, 0], q4)
    T45 = trasl(0.335, 0, 0)  @ rot_about([0, -1, 0], q5)
    T56 = trasl(0.080, 0, 0)  @ rot_about([-1, 0, 0], q6)
    
    '''
    manejar esta parte cuando se agregue la garra
    # tool0 (RPY = pi, -pi/2, 0)
    T6_tool0 = rot_about([1, 0, 0], np.pi) @ rot_about([0, 1, 0], -np.pi/2)
    '''

    T = T01 @ T12 @ T23 @ T34 @ T45 @ T56 #@ T6_tool0
    return T

def jacobian_position(q, delta=0.0001):
    # Alocacion de memoria
    J = np.zeros((3,6))
    # Transformacion homogenea inicial (usando q)
    x = fkine_fanuc(q)[0:3,3] 
    # Iteracion para la derivada de cada columna

    for i in range(6):
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

'''
Lìmites cartesianos (de limits.py)
X: [-0.780, 0.792]
Y: [-0.787, 0.791]
Z: [-0.681, 0.746]
'''
def ikine(xdes, q0, ql, qu):
    epsilon = 1e-5
    max_iter = 1000
    delta = 1e-6
    umbral_cond = 1e3
    k = 1e-2 
    q = copy(q0)
    for i in range(max_iter):
        T = fkine_fanuc(q)
        pos = T[0:3, 3]
        er = xdes - pos
        if np.linalg.norm(er) < epsilon:
            break
        J = jacobian_position(q, delta)
        cond = np.linalg.cond(J)
        if cond < umbral_cond:
            # Jacobiano bien condicionado: usamos pseudoinversa estándar
            J_pinv = np.linalg.pinv(J)
        else:
            # Jacobiano mal condicionado: usar Damped Least Squares
            JT = J.T
            JJT = J @ JT
            amort = k**2 * np.eye(J.shape[0])
            J_pinv = JT @ np.linalg.inv(JJT + amort)
        q = q + np.dot(J_pinv, er)
        q = np.clip(q, ql, qu)
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
