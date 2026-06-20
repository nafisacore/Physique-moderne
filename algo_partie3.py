import numpy as np
import matplotlib.pyplot as plt

def derivee_seconde(f, dx):
    n = len(f)
    d2f = np.zeros(n, dtype=complex)
    d2f[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / dx**2
    d2f[0] = 0
    d2f[-1] = 0
    return d2f

HBAR = 1
MASS = 1

# --- paramètres numériques ---
x_min = -10
x_max = 10
nx = 500
x = np.linspace(x_min, x_max, nx)
dx = x[1] - x[0]

t_max = 3
dt = 0.1 * dx**2
nt = int(t_max / dt)
t = np.linspace(0, t_max, nt)

print("dx =", dx)
print("dt =", dt)
print("nt =", nt)

# --- paquet d'onde initial ---
x0 = -5
k0 = 2
SIGMA = 1

def GaussWP(x, x0, k0, sigma):
    return np.exp(1j*k0*x) * np.exp(-(x-x0)**2 / (2*sigma**2))

# Pas de barrière (V=0)
V = np.zeros(nx)

# --- fonction f(psi) = dpsi/dt selon Schrödinger ---
def f(psi_t, V, dx):
    d2psi = derivee_seconde(psi_t, dx)
    return 1j*HBAR/(2*MASS) * d2psi - 1j*V/HBAR * psi_t

# --- tableau psi : nx lignes, nt colonnes ---
psi = np.zeros((nx, nt), dtype=complex)
psi[:, 0] = GaussWP(x, x0, k0, SIGMA)

# --- propagation RK4 ---
for it in range(1, nt):
    psi_t = psi[:, it-1]

    k1 = f(psi_t, V, dx)
    k2 = f(psi_t + dt/2*k1, V, dx)
    k3 = f(psi_t + dt/2*k2, V, dx)
    k4 = f(psi_t + dt*k3, V, dx)

    psi[:, it] = psi_t + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    psi[0, it] = 0
    psi[-1, it] = 0

# --- graphique ---
plt.plot(x, np.abs(psi[:, 0])**2, label='t=0')
plt.plot(x, np.abs(psi[:, nt//3])**2, label=f't={t[nt//3]:.2f}')
plt.plot(x, np.abs(psi[:, nt//2])**2, label=f't={t[nt//2]:.2f}')
plt.plot(x, np.abs(psi[:, -1])**2, label=f't={t[-1]:.2f}')
plt.xlabel('x')
plt.ylabel('|Ψ|²')
plt.legend()
plt.show()
