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

def runge_kutta_4_adaptive(f, y0, t_span, args=(), tol=1e-15):
    t0, t_end = t_span
    t = [t0]
    y = [y0]
    dt = (t_end - t0) / 10000 # Časový krok

    while t[-1] < t_end:
        current_t = t[-1]
        current_y = y[-1]

        # Požadujeme, aby poslední krok simulace (jedné periody oběhu) v t_end
        if current_t + dt > t_end:
            dt = t_end - current_t

        # Konečně definice Runge-Kuttova algoritmu
        k1 = f(current_y, current_t, *args)
        k2 = f(current_y + k1 * dt / 2, current_t + dt / 2, *args)
        k3 = f(current_y + k2 * dt / 2, current_t + dt / 2, *args)
        k4 = f(current_y + k3 * dt, current_t + dt, *args)
        y_full_step = current_y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Zpřesnění RK metody pomocí zadefinování půlkroku (lepší aproximace chyby)
        dt_half = dt / 2
        k1_half = f(current_y, current_t, *args)
        k2_half = f(current_y + k1_half * dt_half / 2, current_t + dt_half / 2, *args)
        k3_half = f(current_y + k2_half * dt_half / 2, current_t + dt_half / 2, *args)
        k4_half = f(current_y + k3_half * dt_half, current_t + dt_half, *args)
        y_half_step = current_y + (dt_half / 6) * (k1_half + 2 * k2_half + 2 * k3_half + k4_half)

        # Zadefinování druhého půlkroku
        current_y_half_step = y_half_step
        k1_half = f(current_y_half_step, current_t + dt_half, *args)
        k2_half = f(current_y_half_step + k1_half * dt_half / 2, current_t + 3 * dt_half / 2, *args)
        k3_half = f(current_y_half_step + k2_half * dt_half / 2, current_t + 3 * dt_half / 2, *args)
        k4_half = f(current_y_half_step + k3_half * dt_half, current_t + dt, *args)
        y_two_half_steps = current_y_half_step + (dt_half / 6) * (k1_half + 2 * k2_half + 2 * k3_half + k4_half)

        # Odhad naší chyby
        error = np.linalg.norm(y_full_step - y_two_half_steps, ord=np.inf)

        # Algoritmus pro snížení časového kroku pro případ, že je chyba vyšší než naše stanovená (tolerovaná) chyba
        if error > tol:
            dt *= 0.5
        else:
            t.append(current_t + dt)
            y.append(y_two_half_steps)
            if error < tol / 4:
                dt *= 2  # Zvýšíme časový krok pro případ, že by chyba byla mnohem menší než tolerovaná (zde násobíme 2, výše naopak dělíme)

    return np.array(t), np.array(y)

# Integrační parametry
T = 2 * np.pi
t_span = (0, T)
tol = 1e-15  # Maximální chyba, kterou tolerujeme (podlé té se náš časový krok posléze mění)

# Algoritmus pro vyřešení dif. rovnic
t, states = runge_kutta_4_adaptive(equations_of_motion, state0, t_span, tol=tol)

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

# Formality k zobrazení
plt.subplot(1, 2, 1)
plt.scatter([0], [0], color="black", label="Cenrální těleso")
plt.plot(states[:, 0], states[:, 1])
plt.title("Trajektorie částice v centrálním poli")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")

# Zobrazení grafů zachování
plt.subplot(1, 2, 2)
plt.plot(t, energies, label="Hamiltonián")
plt.plot(t, momenta, label="Moment hybnosti")
plt.title("Zákony zachování")
plt.xlabel("Čas")
plt.legend()

plt.tight_layout()
plt.show()
