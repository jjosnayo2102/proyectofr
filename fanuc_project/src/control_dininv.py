#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from markers import *
from proyfunctions import *
from roslib import packages
import rbdl


rospy.init_node("control_pdg")
pub = rospy.Publisher('joint_states', JointState, queue_size=1000)
bmarker_actual  = BallMarker(color['RED'])
bmarker_deseado = BallMarker(color['GREEN'])
# Archivos donde se almacenara los datos
fqact = open("qactual2.txt", "w")
fqdes = open("qdeseado2.txt", "w")
fxact = open("xactual2.txt", "w")
fxdes = open("xdeseado2.txt", "w")

# Nombres de las articulaciones
jnames = ['joint_1', 'joint_2','joint_3','joint_4','joint_5','joint_6','gripper_base-finger_1','gripper_base-finger_2']
# Objeto (mensaje) de tipo JointState
jstate = JointState()
# Valores del mensaje
jstate.header.stamp = rospy.Time.now()
jstate.name = jnames

# =============================================================
# Configuracion articular inicial (en radianes)
q = np.array([0.0, 0, 0, 0, 0, 0, 0, 0])
# Velocidad inicial
dq = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0])
# Aceleracion inicial
ddq = np.array([0., 0., 0., 0., 0., 0., 0.,0.])
# Configuracion articular deseada
qdes = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0])
# Velocidad articular deseada
dqdes = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0])
# Aceleracion articular deseada
ddqdes = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0])
# =============================================================

# Posicion resultante de la configuracion articular deseada
xdes = fkine_fanuc(qdes)[0:3,3]
# Copiar la configuracion articular en el mensaje a ser publicado
jstate.position = q
pub.publish(jstate)

# Modelo RBDL
modelo = rbdl.loadModel('../urdf/fanuc125.urdf')
ndof   = modelo.q_size     # Grados de libertad
zeros = np.zeros(ndof)     # Vector de ceros

# Frecuencia del envio (en Hz)
freq = 50
dt = 1.0/freq
rate = rospy.Rate(freq)

# Simulador dinamico del robot
robot = Robot(q, dq, ndof, dt)

# Bucle de ejecucion continua
t = 0.0

# Se definen las ganancias del controlador
valores = 0.1*np.array([5, 5, 5])
Kp = np.diag(valores)
Kd = 2*np.sqrt(Kp)

J_PAS=np.zeros([3, ndof])
epas = np.array([0,0,0])
b2 = np.zeros(ndof)          # Para efectos no lineales
M2 = np.zeros([ndof, ndof])  # Para matriz de inercia
while not rospy.is_shutdown():

    # Leer valores del simulador
    q  = robot.read_joint_positions()
    dq = robot.read_joint_velocities()
    if qdes[0]<0:
      q[0] = 0
    if qdes[3]>=0.8:
      qdes[3]=0.8
    if qdes[4]>=0.5:
      qdes[4] = 0.5
    if qdes[4]<=-0.5:
      qdes[4] = -0.5
    if qdes[5]>=0.5:
      qdes[5] = 0.5
    if qdes[5]<=-0.5:
      qdes[5] = -0.5
    if qdes[6]<=0:
      qdes[6] = 0
    if qdes[7]>=0:
      q[7] = 0
    print(q)
    
    # Posicion actual del efector final
    x = fkine_fanuc(q)[0:3,3]
    # Tiempo actual (necesario como indicador para ROS)
    jstate.header.stamp = rospy.Time.now()

    # Almacenamiento de datos
    fxact.write(str(t)+' '+str(x[0])+' '+str(x[1])+' '+str(x[2])+'\n')
    fxdes.write(str(t)+' '+str(xdes[0])+' '+str(xdes[1])+' '+str(xdes[2])+'\n')
    fqact.write(str(t)+' '+str(q[0])+' '+str(q[1])+' '+ str(q[2])+' '+ str(q[3])+' '+str(q[4])+' '+str(q[5])+'\n ')
    fqdes.write(str(t)+' '+str(qdes[0])+' '+str(qdes[1])+' '+ str(qdes[2])+' '+ str(qdes[3])+' '+str(qdes[4])+' '+str(qdes[5])+'\n ')

    # ----------------------------
    # Control dinamico (COMPLETAR)
    # ----------------------------
    rbdl.CompositeRigidBodyAlgorithm(modelo, q ,M2)
    rbdl.NonlinearEffects(modelo,q,dq , b2)
    J = jacobian_position(q,dt)
    dJ = (J - J_PAS)/dt
    e = xdes-x
    de = (e-epas)/dt
    print('d_error')
    print(de)
    u = M2.dot(np.linalg.pinv(J)).dot(-dJ.dot(dq)+Kd.dot(de)+Kp.dot(e))+b2
    epas = copy(e)
    J_PAS =copy(J)


    # Simulacion del robot
    robot.send_command(u)

    # Publicacion del mensaje
    jstate.position = q
    pub.publish(jstate)
    bmarker_deseado.xyz(xdes)
    bmarker_actual.xyz(x)
    t = t+dt
    # Esperar hasta la siguiente  iteracion
    rate.sleep()

fqact.close()
fqdes.close()
fxact.close()
fxdes.close()
