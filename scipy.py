import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

epsilon = 0.7
x0 = 1 - epsilon
y0 = 0
px0 = 0
if epsilon == 1:
    py0 = 1
else:
    py0 = np.sqrt((1 + epsilon) / (1 - epsilon))

def potential(x, y):
  r = np.sqrt(x**2 + y**2)
  return -1/r

def hamiltonian(x, y, px, py,):
  return 0.5*(px**2 + py**2) + potential(x,y)

def angular_momentum(x, y, px, py):
  return x*py - y*px

def equations_of_motion(state, t):
  x, y, px, py = state

  dxdt = px
  dydt = py
  dpxdt = - x/np.sqrt((x**2 + y**2)**3)
  dpydt = - y/np.sqrt((x**2 + y**2)**3)

  return[dxdt, dydt, dpxdt, dpydt]

t = np.linspace(0, 2*np.pi, 1000)

initial_state = [x0, y0, px0, py0]

solution = odeint(equations_of_motion, initial_state, t)

x, y, px, py = solution.T

final_state = solution[-1]  # Assuming t[-1] is close to one period

position_difference = np.linalg.norm(final_state[:2] - initial_state[:2])

print(f"Rozdíl v poloze po jedné periodě: {position_difference:.2e}")

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.scatter([0], [0], color="black", label="Centrální těleso")
plt.title('Trajektorie')
plt.axis('equal')
plt.show()
