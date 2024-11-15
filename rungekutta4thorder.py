import numpy as np
import matplotlib.pyplot as plt

epsilon = 0.9
x0 = 1 - epsilon
y0 = 0
px0 = 0
py0 = np.sqrt((1 + epsilon) / (1 - epsilon)) if epsilon != 1 else 1
state0 = np.array([x0, y0, px0, py0])

def potential(x, y):
    r = np.sqrt(x**2 + y**2)
    return -1 / r

def equations_of_motion(state, t):
    x, y, px, py = state
    dxdt = px
    dydt = py
    dpxdt = -x/np.sqrt((x**2 + y**2)**3)
    dpydt = -y/np.sqrt((x**2 + y**2)**3)
    return np.array([dxdt, dydt,  dpxdt,  dpydt])

def runge_kutta_4(f, state, t, dt):
    k1 = dt * f(state, t)
    k2 = dt * f(state + k1 / 2, t + dt / 2)
    k3 = dt * f(state + k2 / 2, t + dt / 2)
    k4 = dt * f(state + k3, t + dt)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

dt = 0.01 
T = 2 * np.pi  
num_steps = int(T / dt)  

states = [state0]
times = [0]

for step in range(num_steps):
    next_state = runge_kutta_4(equations_of_motion, states[-1], times[-1], dt)
    states.append(next_state)
    times.append(times[-1] + dt)

states = np.array(states)

def hamiltonian(state):
    x, y, px, py = state
    return 0.5*(px**2 + py**2) + potential(x, y)

def angular_momentum(state):
    x, y, px, py = state
    return x * py - y * px

energies = np.array([hamiltonian(s) for s in states])
momenta = np.array([angular_momentum(s) for s in states])

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter([0], [0], color="red", label="Centrální těleso")
plt.plot(states[:, 0], states[:, 1])
plt.title("Trajektorie částice")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")

plt.subplot(1, 2, 2)
plt.plot(times, energies, label="Energie")
plt.plot(times, momenta, label="Moment hybnosti")
plt.title("Zákony zachování")
plt.xlabel("Čas")
plt.legend()

plt.tight_layout()
plt.show()
