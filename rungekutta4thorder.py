import numpy as np
import matplotlib.pyplot as plt

# Počáteční podmínky (dle zadání)
epsilon = 0.9
x0 = 1 - epsilon
y0 = 0
px0 = 0
# Podmínka odpovídající zadání pro případ epsilon = 1 tak, aby nedocházelo k dělení nulou
py0 = np.sqrt((1 + epsilon) / (1 - epsilon)) if epsilon != 1 else 1
state0 = np.array([x0, y0, px0, py0])

# Zadefinování charakterizujích veličin
def potential(x, y):
    r = np.sqrt(x**2 + y**2)
    return -1 / r
    
def hamiltonian(state):
    x, y, px, py = state
    return 0.5*(px**2 + py**2) + potential(x, y)

def angular_momentum(state):
    x, y, px, py = state
    return x * py - y * px

# Soustava rovnic pohybu
def equations_of_motion(state, t):
    x, y, px, py = state
    dxdt = px
    dydt = py
    dpxdt = -x/np.sqrt((x**2 + y**2)**3)
    dpydt = -y/np.sqrt((x**2 + y**2)**3)
    return np.array([dxdt, dydt,  dpxdt,  dpydt])

# Zadefinování Runge-Kuttova algoritmu 4. řádu
def runge_kutta_4(f, state, t, dt):
    k1 = dt * f(state, t)
    k2 = dt * f(state + k1 / 2, t + dt / 2)
    k3 = dt * f(state + k2 / 2, t + dt / 2)
    k4 = dt * f(state + k3, t + dt)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Časový krok, perioda, počet kroků integrace
dt = 0.0001 
T = 2 * np.pi  
num_steps = int(T / dt)  

states = [state0]
times = [0]

# Vyřešení soustavy ODR pro každý krok
for step in range(num_steps):
    next_state = runge_kutta_4(equations_of_motion, states[-1], times[-1], dt)
    states.append(next_state)
    times.append(times[-1] + dt)

# Vytvoření "seznamu" hodnot energie (Hamiltoniánu) a momentu hybnosti pro ověření jejich zachování
states = np.array(states)
energies = np.array([hamiltonian(s) for s in states])
momenta = np.array([angular_momentum(s) for s in states])

# Rozdíl polohy částice v časech t = 0 a t = T = 2π 
final_state = states[-1]
position_difference = np.linalg.norm(final_state[:2] - state0[:2])
print(f"Rozdíl v poloze po jedné periodě: {position_difference:.2e}")

# Znázornění trajektorie částice v centrálním poli
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter([0], [0], color="red", label="Centrální těleso")
plt.plot(states[:, 0], states[:, 1])
plt.title("Trajektorie částice")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")

# Znázornění zachovávajícího se Hamiltoniánu (energie) a momentu hybnosti
plt.subplot(1, 2, 2)
plt.plot(times, energies, label="Energie")
plt.plot(times, momenta, label="Moment hybnosti")
plt.title("Zákony zachování")
plt.xlabel("Čas")
plt.legend()

plt.tight_layout()
plt.show()
