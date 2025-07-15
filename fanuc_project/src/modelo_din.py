import rbdl
import numpy as np
import subprocess
import tempfile
from urdf_parser_py.urdf import URDF

def cargar_urdf_desde_xacro(xacro_path):
    xacro_result = subprocess.run(['xacro', xacro_path], capture_output=True, text=True)
    if xacro_result.returncode != 0:
        print("Error procesando xacro:\n" + xacro_result.stderr)
        exit(1)
    return xacro_result.stdout  # Devolver string XML

if __name__ == '__main__':
    xacro_path = '/home/jjosnayo/lab_ws/src/proyectofr/fanuc_robot/fanuc_lrmate200id_support/urdf/lrmate200id.xacro'
    
    # Generar string del URDF
    urdf_xml = cargar_urdf_desde_xacro(xacro_path)

    # Escribir URDF a archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".urdf") as urdf_file:
        urdf_file.write(urdf_xml.encode('utf-8'))
        urdf_path = urdf_file.name  # Ruta temporal al archivo

    # Cargar modelo RBDL
    modelo = rbdl.loadModel(urdf_path)
    ndof = modelo.q_size

    # Configuración articular de ejemplo
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dq = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0])
    ddq = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


    zeros = np.zeros(ndof)
    tau = np.zeros(ndof)
    g = np.zeros(ndof)
    c = np.zeros(ndof)
    mi = np.zeros(ndof)
    M = np.zeros((ndof, ndof))
    e = np.eye(ndof)

    # Torque general
    rbdl.InverseDynamics(modelo, q, dq, ddq, tau)
    print('Vector de torques obtenidos con la función InverseDynamics')
    print(np.round(tau, 5))

    # Gravedad
    rbdl.InverseDynamics(modelo, q, zeros, zeros, g)
    g = np.round(g, 5)
    print('MATRIZ GRAVEDAD')
    print(np.round(g, 3))

    # Coriolis + centrífuga
    rbdl.InverseDynamics(modelo, q, dq, zeros, c)
    c = c - g
    c = np.round(c, 2)
    print('MATRIZ F y C')
    print(c)

    # Matriz de inercia
    for i in range(ndof):
        rbdl.InverseDynamics(modelo, q, zeros, e[i, :], mi)
        M[i, :] = mi - g
    print('MATRIZ INERCIA')
    print(np.round(M, 4))

    # Verificación dinámica
    tau2 = M @ ddq + c + g
    print('Vector de torques obtenidos operando matrices')
    print(np.round(tau2, 5))


