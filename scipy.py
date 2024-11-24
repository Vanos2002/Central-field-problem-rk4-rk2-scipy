import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Definování počátečních podmínek (dle zadání)
epsilon = 0.7
x0 = 1 - epsilon
y0 = 0
px0 = 0
# Podmínka pro poč. podmínku při epsilon = 1, jinak bychom dělili nulou
if epsilon == 1:
    py0 = 1
else:
    py0 = np.sqrt((1 + epsilon) / (1 - epsilon))

# Zadefinování vystupujících/charakterizujících veličin
def potential(x, y):
  r = np.sqrt(x**2 + y**2)
  return -1/r

def hamiltonian(x, y, px, py,):
  return 0.5*(px**2 + py**2) + potential(x,y)

def angular_momentum(x, y, px, py):
  return x*py - y*px
    
# Soustava rovnic pohybu
def equations_of_motion(state, t):
  x, y, px, py = state

  dxdt = px
  dydt = py
  dpxdt = - x/np.sqrt((x**2 + y**2)**3)
  dpydt = - y/np.sqrt((x**2 + y**2)**3)

  return[dxdt, dydt, dpxdt, dpydt]

# Časový krok integrace, počet kroků jsme stanovili jako 2π*1E+4, abychom mohli srovnávat s RK4 a RK2
t = np.linspace(0, 2*np.pi, 62832)

# "Vektor" počátečních podmínek
initial_state = [x0, y0, px0, py0]

solution = odeint(equations_of_motion, initial_state, t)

x, y, px, py = solution.T

# Rozdíl polohy částice v časech t = 0 a t = T = 2π 
final_state = solution[-1] 
position_difference = np.linalg.norm(final_state[:2] - initial_state[:2])
print(f"Rozdíl v poloze po jedné periodě oběhu: {position_difference:.2e}")

# Vykreslení trajektorie částice
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.scatter([0], [0], color="black", label="Centrální těleso")
plt.title('Trajektorie')
plt.axis('equal')
plt.show()
