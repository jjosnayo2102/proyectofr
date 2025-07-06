import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === TU FKINE ===
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
    T01 = trasl(0, 0, 0)      @ rot_about([0, 0, 1], q1)
    T12 = trasl(0.050, 0, 0)  @ rot_about([0, 1, 0], q2)
    T23 = trasl(0, 0, 0.330)  @ rot_about([0, -1, 0], q3)
    T34 = trasl(0, 0, 0.035)  @ rot_about([-1, 0, 0], q4)
    T45 = trasl(0.335, 0, 0)  @ rot_about([0, -1, 0], q5)
    T56 = trasl(0.080, 0, 0)  @ rot_about([-1, 0, 0], q6)
    T6_tool0 = rot_about([1, 0, 0], np.pi) @ rot_about([0, 1, 0], -np.pi/2)
    return T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T6_tool0

# === LÍMITES ARTICULARES ===
ql = np.array([-2.967, -1.745, -1.222, -3.316, -2.181, -6.283])
qu = np.array([ 2.967,  2.530,  3.577,  3.316,  2.181,  6.283])

# === MUESTREO ALEATORIO DE CONFIGURACIONES ===
n_samples = 100000
qs = np.random.uniform(low=ql, high=qu, size=(n_samples, 6))

# === EVALUAR POSICIONES CARTESIANAS ===
positions = []
for q in qs:
    T = fkine_fanuc(q)
    p = T[:3, 3]  # coordenadas (x, y, z)
    positions.append(p)

positions = np.array(positions)

# === OBTENER LÍMITES CARTESIANOS ===
x_min, x_max = positions[:,0].min(), positions[:,0].max()
y_min, y_max = positions[:,1].min(), positions[:,1].max()
z_min, z_max = positions[:,2].min(), positions[:,2].max()

print("Límites cartesianos aproximados del efector final:")
print(f"X: [{x_min:.3f}, {x_max:.3f}]")
print(f"Y: [{y_min:.3f}, {y_max:.3f}]")
print(f"Z: [{z_min:.3f}, {z_max:.3f}]")

# === GRAFICAR EL ESPACIO ALCANZABLE ===
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:,0], positions[:,1], positions[:,2], s=0.1, alpha=0.5)
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('Workspace del robot (aproximado)')
plt.tight_layout()
plt.show()

