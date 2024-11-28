import numpy as np
import matplotlib.pyplot as plt

# Počáteční podmínky (dle zadání)
epsilon = 0.7
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

# Zadefinování adaptivní RK2 metody
def runge_kutta_2_adaptive(f, y0, t_span, args=(), tol=1e-15):
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

        # Celý krok zadefinovaný pomocí RK2
        k1 = f(current_y, current_t, *args)
        k2 = f(current_y + dt * k1/2, current_t + dt/2, *args)
        y_full_step = current_y + dt * k2

        # 1. půlrok
        dt_half = dt/2 # Zadefinování poloviny časového kroku
        k1_half = f(current_y, current_t, *args)
        k2_half = f(current_y + dt_half * k1_half/2, current_t + dt_half/2, *args)
        y_half_step = current_y + dt_half * k2_half

        # 2.půlkrok
        k1_half = f(y_half_step, current_t + dt_half, *args)
        k2_half = f(y_half_step + dt_half * k1_half/2, current_t + dt_half + dt_half/2, *args)
        y_two_half_steps = y_half_step + dt_half * k2_half
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
tol = 1e-15  # Maximální chyba, kterou tolerujeme (podlé té se náš časovýž krok posléze mění)

# Algoritmus pro vyřešení dif. rovnic
t, states = runge_kutta_2_adaptive(equations_of_motion, state0, t_span, tol=tol)

# Vytvoření "seznamu" hodnot energie (Hamiltoniánu) a momentu hybnosti pro ověření jejich zachování
energies = np.array([hamiltonian(s) for s in states])
momenta = np.array([angular_momentum(s) for s in states])

# Rozdíl polohy částice v časech t = 0 a t = T = 2π 
final_state = states[-1]
position_difference = np.linalg.norm(final_state[:2] - state0[:2])
print(f"Rozdíl v poloze po jedné periodě: {position_difference:.2e}")

# Znázornění trajektorie částice v centrálním poli
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter([0], [0], color="black", label="Centrální těleso")
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
