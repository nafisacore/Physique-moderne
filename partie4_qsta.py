import numpy as np
import matplotlib.pyplot as plt
# Fonctions de base
def derivee_seconde(f, dx):
    n = len(f)
    d2f = np.zeros(n, dtype=complex)
    d2f[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / dx**2
    d2f[0] = 0
    d2f[-1] = 0
    return d2f
def GaussWP(x, t, x0, k0, a, hbar, m):
    X = x - x0
    denom = m * a**2 + 2j * hbar * t
    prefacteur = (1 / (8 * np.pi**3))**0.25
    racine = np.sqrt(4 * np.pi * m * a / denom)
    exposant = (m / 4) * ((a**2 * k0 + 2j * X)**2 / denom - (a**2 * k0**2) / 4)
    return prefacteur * racine * np.exp(exposant)
HBAR = 1
MASS = 1

#grille
x_min = -20
x_max = 20
nx = 1000
x = np.linspace(x_min, x_max, nx)
dx = x[1] - x[0]
t_max = 5
dt = 0.2 * dx**2
nt = int(t_max / dt)
t = np.linspace(0, t_max, nt)
print("nt =", nt)

#paramètres du paquet d'onde initial
x0 = -5
k0 = 2
a = 1

#barrière de potentiel
V0 = 0
V = np.zeros(nx)
for i in range(nx):
    if 0 <= x[i] <= 5:
        V[i] = V0
def f(psi_t, V, dx):
    d2psi = derivee_seconde(psi_t, dx)
    return 1j * HBAR / (2 * MASS) * d2psi - 1j * V / HBAR * psi_t
  
# Evolution du paquet d'ondes libre (V0=0) et calcul de tau_0,num
psi = np.zeros((nx, nt), dtype=complex)
psi[:, 0] = GaussWP(x, 0, x0, k0, a, HBAR, MASS)
for it in range(1, nt):
    psi_t = psi[:, it-1]
    k1 = f(psi_t, V, dx)
    k2 = f(psi_t + dt/2 * k1, V, dx)
    k3 = f(psi_t + dt/2 * k2, V, dx)
    k4 = f(psi_t + dt * k3, V, dx)
    psi[:, it] = psi_t + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    psi[0, it] = 0
    psi[-1, it] = 0

x_max_pic = x[np.argmax(np.abs(psi)**2, axis=0)]
tau_0_num = t[np.argmax(x_max_pic >= x_max_pic[0] + 5)]
print(f"tau_0,num = {tau_0_num:.5f}")

# --- affichage ---
plt.figure(figsize=(9, 5))
plt.plot(x, np.abs(psi[:, 0])**2, label='t=0')
plt.plot(x, np.abs(psi[:, nt//2])**2, label=f't={t[nt//2]:.2f}')
plt.plot(x, np.abs(psi[:, -1])**2, label=f't={t[-1]:.2f}')
plt.xlabel('x')
plt.ylabel('|Ψ|²')
plt.title("Évolution du paquet d'ondes libre (V0=0)")
plt.legend()
plt.show()
