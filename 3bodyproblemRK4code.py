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
    ps, pz, pm = state[:3], state[3:6], state[6:9]  # Výběr složek pozice pro daná tělesa
    vs, vz, vm = state[9:12], state[12:15], state[15:18]  # Výběr složek rychlosti pro daná tělesa

    # Spočtení zrychlení dle vztahů, které jsme uvedli výše
    accelerations_array = accelerations(ps, pz, pm)
    as_, az, am = accelerations_array

    # Zahrnutí rychlostí a zrychlení do jednoho vektoru
    derivatives = np.concatenate([vs, vz, vm, as_, az, am])
    return derivatives

# Integrační kroky
dt = 100 # Časový krok v sekundách (odpovídající jedné hodině)
t_max = 365.25 * 24 * 3600  # Perioda jednoho oběhu Země kolem Slunce, odpovída asi pi*10^7 s
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

def total_energy(state):
    ps, pz, pm = state[:3], state[3:6], state[6:9]
    vs, vz, vm = state[9:12], state[12:15], state[15:18]

    r_s_z = np.linalg.norm(ps - pz)
    r_s_m = np.linalg.norm(ps - pm)
    r_z_m = np.linalg.norm(pz - pm)
    
    # Definice kinetické energie soustavy
    KE = 0.5 * M_s * np.linalg.norm(vs)**2 + 0.5 * M_z * np.linalg.norm(vz)**2 + 0.5 * M_m * np.linalg.norm(vm)**2
    
    # Definice potenciální energie soustavy
    PE = -G * (M_s * M_z / r_s_z + M_s * M_m / r_s_m + M_z * M_m / r_z_m)
    
    return KE + PE

energy_values = []

for i in range(1, n_steps):
    # "Aktualizování" pomocí RK4
    trajectories[i] = runge_kutta_4(dynamics, trajectories[i - 1], time[i - 1], dt)
    
    # Spočítá energii pro konkrétní krok
    E = total_energy(trajectories[i])
    
    # "Skladování" hodnot
    energy_values.append(E)

# Převod na vektor pro jednodušší manipulaci
energy_values = np.array(energy_values)

def angular_momentum(state):
    positions = state[:9].reshape(3, 3)  # Vybrání pozic ze stavu (rozmístění do 3 kategorií po 3 třech položkách, pro Slunce, Zemi a Měsíc)
    velocities = state[9:].reshape(3, 3)  # Vybrání rychlostí ze stavu (rozmístění do 3 kategorií po 3 třech položkách, pro Slunce, Zemi a Měsíc)
    masses = np.array([M_s, M_z, M_m])  # Zapsání jednotlivých hmotností jako vektor
    
    # Spočtení těžiště
    total_mass = np.sum(masses)
    center_of_mass = np.sum(positions.T * masses, axis=1) / total_mass
    
    # Výpočet relativní pozice vzhledem k těžišti
    relative_positions = positions - center_of_mass
    
    # Konečně výpočet momentu hybnosti
    L = np.sum(np.cross(relative_positions, masses[:, None] * velocities), axis=0)
    return L

angular_momentum_values = []

for i in range(1, n_steps):
    trajectories[i] = runge_kutta_4(dynamics, trajectories[i - 1], time[i - 1], dt)
    L = angular_momentum(trajectories[i])
    angular_momentum_values.append(L)

# Převod na vektor pro jednodušší manipulaci
angular_momentum_values = np.array(angular_momentum_values)

# Zjištění pro zachování celkové energie
initial_energy_value = energy_values[0]
final_energy_value = energy_values[-1]
differenceE = np.linalg.norm(final_energy_value - initial_energy_value)

# Zakomponovali jsme jednotky, jelikož jsme udávali rychlost v km*s^-1 a gravitační konstantu v km^3*kg^-1s^-2 a polohu v km, bude hodnotu energie 10^6 větší, proto MJ
print(f"Počáteční energie soustavy: {np.linalg.norm(initial_energy_value):.2e} MJ")
print(f"Energie soustavy po jednom oběhu Země kolem Slunce: {np.linalg.norm(final_energy_value):.2e} MJ")
print(f"Rodíl energií: {differenceE:.2e} MJ")

# Zjištění pro zachování momentu hybnosti
initial_angular_momentum = angular_momentum_values[0]
final_angular_momentum = angular_momentum_values[-1]
differenceL = np.linalg.norm(final_angular_momentum - initial_angular_momentum)

# Zakomponovali jsme jednotky, jelikož jsme udávali rychlost v km*s^-1, polohu v km a hmotnost v kg, je proto moment hybnosti v kg*km^2*s^-1
print(f"Počáteční moment hybnosti: {np.linalg.norm(initial_angular_momentum):.2e} kg*km^2*s^-1")
print(f"Moment hybnosti po jednom oběhu Země kolem Slunce: {np.linalg.norm(final_angular_momentum):.2e} kg*km^2*s^-1")
print(f"Rozdíl momentu hybnosti: {differenceL:.2e} kg*km^2*s^-1")

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
