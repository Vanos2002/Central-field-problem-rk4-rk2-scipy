import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Konstanty
G = 6.67E-20 # Gravitační konstanta v jednotkách km^3 kg^-1 s^-2
AU = 1.496E+8 # Astronomická jednotka v km
Ld = 388400 # Vzdálenost Měsíce od Země (z angličtiny lunar distance)

# Hmotnost těles je uvedena v kg
M_s = 1.99E+30 # Slunce
M_z = 5.97E+24  # Země
M_m = 7.35E+22  # Měsíc

# Počáteční souřadnice a vektor rychlosti Slunce, jednotky souřadnic jsou km a rychlostí pak km s^-1
ps_0 = np.array([0, 0, 0])
vs_0 = np.array([0, 0, 0])

# Počáteční souřadnice a vektor rychlosti Země. Opět jednotky souřadnic jsou km a rychlostí pak km s^-1
pz_0 = np.array([AU, 0, 0])
vz_0 = np.array([0, 29.78, 0])

# Počáteční souřadnice a vektor rychlosti Měsíce. Opět jednotky souřadnic jsou km a rychlostí pak km s^-1
pm_0 = np.array([AU + Ld, 0, 0])
vm_0 = np.array([0, 30.8, 0])

# Zadefinování rovnic pohybu pro zrychlení
def accelerations(ps, pz, pm):

      a_s = -G*M_z*(ps - pz)/(np.linalg.norm(ps - pz)**3) - G*M_m*(ps - pm)/(np.linalg.norm(ps - pm)**3)

      a_z = -G*M_m*(pz - pm)/(np.linalg.norm(pz - pm)**3) - G*M_s*(pz - ps)/(np.linalg.norm(pz - ps)**3)

      a_m = -G*M_s*(pm - ps)/(np.linalg.norm(pm - ps)**3) - G*M_z*(pm - pz)/(np.linalg.norm(pm - pz)**3)

      return np.array([a_s, a_z, a_m])

# Zavolání počátečního vektoru
state_0 = np.concatenate([ps_0, pz_0, pm_0, vs_0, vz_0, vm_0])

# Zadefinování Runge-Kuttova algoritmu 4. řádu
def runge_kutta_4(f, state, t, dt):
    k1 = dt * f(state, t)
    k2 = dt * f(state + k1 / 2, t + dt / 2)
    k3 = dt * f(state + k2 / 2, t + dt / 2)
    k4 = dt * f(state + k3, t + dt)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def dynamics(state, t):
    ps, pz, pm = state[:3], state[3:6], state[6:9]  # Extrahování složek pozice pro daná tělesa
    vs, vz, vm = state[9:12], state[12:15], state[15:18]  # Extrahování složek rychlosti pro daná tělesa

    # Spočtení zrychlení dle vztahů, které jsme uvedli výše
    accelerations_array = accelerations(ps, pz, pm)
    as_, az, am = accelerations_array

    # Zahrnutí rychlostí a zrychlení do jednoho vektoru
    derivatives = np.concatenate([vs, vz, vm, as_, az, am])
    return derivatives

# Integrační kroky
dt = 3600 # Časový krok v sekundách (odpovídající jedné hodině)
t_max = 365 * 24 * 3600  # Perioda jednoho oběhu Země kolem Slunce (uvedli jsme pro nepřestupný rok)
n_steps = int(t_max / dt)  # Počet kroků (odpovídá tedy počtu hodin v nepřestupném roku)
time = np.linspace(0, t_max, n_steps) # Čas (od, do, přes kolik kroků)

# Vypsání trajektoriíí
trajectories = np.zeros((n_steps, len(state_0)))
trajectories[0] = state_0  # Počáteční stav

# Probíhání simulace (zde konečně implementujeme RK4 :))
for i in range(1, n_steps):
    trajectories[i] = runge_kutta_4(dynamics, trajectories[i - 1], time[i - 1], dt)

# Výběr jednotlivých složek vektorů pozic
positions_sun = trajectories[:, :3]
positions_earth = trajectories[:, 3:6]
positions_moon = trajectories[:, 6:9]

# Vykreslení výsledků
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Vykreslení trajektorií našich tří těles
ax.plot(positions_sun[:, 0], positions_sun[:, 1], positions_sun[:, 2], label="Trajektorie Slunce", color="orange")
ax.plot(positions_earth[:, 0], positions_earth[:, 1], positions_earth[:, 2], label="Trajektorie Země", color="red")
ax.plot(positions_moon[:, 0], positions_moon[:, 1], positions_moon[:, 2], label="Trajektorie Měsíce", color="white")

# Uvedení polohy tělesa dle počátečních podmínek, které jsme uvedli výše
ax.scatter(ps_0[0], ps_0[1], ps_0[2], color="yellow", s=100, label="Počáteční poloha Slunce")
ax.scatter(pz_0[0], pz_0[1], pz_0[2], color="blue", s=50, label="Počáteční poloha Země")
ax.scatter(pm_0[0], pm_0[1], pm_0[2], color="gray", s=25, label="Počáteční poloha Měsíce")

# Popis os s legendou a názvem naší simulace
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.legend()
ax.set_title("Simulace problému tří těles: Slunce-Země-Měsíc")
plt.show()
