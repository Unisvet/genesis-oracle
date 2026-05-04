import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def simulate_systems():
    # ==========================================
    # 1. Radioactive Decay
    # dN/dt = -lambda * N
    # Dimensionless: dN'/dtau = -N'
    # where N' = N/N0, tau = lambda * t
    # ==========================================
    
    # Parameters
    lam = 0.1 # Decay constant (1/s)
    N0 = 1000 # Initial number of atoms
    
    # Standard System
    def decay_standard(t, n_arr):
        N = n_arr[0]
        dNdt = -lam * N
        return [dNdt]
        
    t_span_standard = (0, 50)
    t_eval_standard = np.linspace(t_span_standard[0], t_span_standard[1], 300)
    sol_decay = solve_ivp(decay_standard, t_span_standard, [N0], t_eval=t_eval_standard)
    
    # Dimensionless System
    def decay_dimensionless(tau, n_prime_arr):
        N_prime = n_prime_arr[0]
        dN_prime_dtau = -N_prime
        return [dN_prime_dtau]
        
    tau_span = (0, lam * 50) # tau = lambda * t
    tau_eval = np.linspace(tau_span[0], tau_span[1], 300)
    sol_decay_dimless = solve_ivp(decay_dimensionless, tau_span, [1.0], t_eval=tau_eval)

    # Plot Decay
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Physical Plot
    axs[0].plot(sol_decay.t, sol_decay.y[0], label="N(t)", color="red")
    axs[0].set_title("Radioaktiver Zerfall (Physikalische Einheiten)")
    axs[0].set_xlabel("Zeit t [s]")
    axs[0].set_ylabel("Anzahl Atome N")
    axs[0].grid(True)
    axs[0].legend()
    
    # Dimensionless Plot
    axs[1].plot(sol_decay_dimless.t, sol_decay_dimless.y[0], label="N'(tau)", color="orange")
    axs[1].set_title("Radioaktiver Zerfall (Dimensionslos)")
    axs[1].set_xlabel("Skalierte Zeit $\\tau = \\lambda t$")
    axs[1].set_ylabel("Skalierte Anzahl N' = N/N0")
    axs[1].grid(True)
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

    # ==========================================
    # 2. Pendulum
    # d^2 theta / dt^2 + (g/L)*sin(theta) = 0
    # First Order System:
    # d theta / dt = omega
    # d omega / dt = -(g/L)*sin(theta)
    # 
    # Dimensionless:
    # d^2 theta / dtau^2 + sin(theta) = 0
    # where tau = t * sqrt(g/L)
    # ==========================================
    
    # Parameters
    g = 9.81
    L = 1.0
    theta0 = np.pi / 4 # 45 degrees
    omega0 = 0.0
    
    # Standard System
    def pendulum_standard(t, state):
        theta, omega = state
        dtheta_dt = omega
        domega_dt = -(g/L) * np.sin(theta)
        return [dtheta_dt, domega_dt]
        
    t_span_pendulum = (0, 10)
    t_eval_pendulum = np.linspace(0, 10, 500)
    sol_pend = solve_ivp(pendulum_standard, t_span_pendulum, [theta0, omega0], t_eval=t_eval_pendulum)
    
    # Dimensionless System
    def pendulum_dimensionless(tau, state):
        theta, omega_prime = state
        dtheta_dtau = omega_prime
        domega_prime_dtau = -np.sin(theta)
        return [dtheta_dtau, domega_prime_dtau]

    omega0_prime = omega0 / np.sqrt(g/L) # Dimensionless initial angular velocity
    tau_span_pendulum = (0, 10 * np.sqrt(g/L))
    tau_eval_pendulum = np.linspace(tau_span_pendulum[0], tau_span_pendulum[1], 500)
    sol_pend_dimless = solve_ivp(pendulum_dimensionless, tau_span_pendulum, [theta0, omega0_prime], t_eval=tau_eval_pendulum)

    # Plot Pendulum
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))
    
    # Standard Plot
    axs2[0].plot(sol_pend.t, sol_pend.y[0], label="Winkel $\\theta(t)$", color="blue")
    axs2[0].plot(sol_pend.t, sol_pend.y[1], label="Winkelgeschwindigkeit $\\dot{\\theta}(t)$", color="cornflowerblue", linestyle="--")
    axs2[0].set_title("Harmonisches Pendel (Physikalische Einheiten)")
    axs2[0].set_xlabel("Zeit t [s]")
    axs2[0].set_ylabel("Winkel / Winkelgeschwindigkeit")
    axs2[0].grid(True)
    axs2[0].legend(loc="upper right")
    
    # Dimensionless Plot
    axs2[1].plot(sol_pend_dimless.t, sol_pend_dimless.y[0], label="Winkel $\\theta(\\tau)$", color="green")
    axs2[1].plot(sol_pend_dimless.t, sol_pend_dimless.y[1], label="Skalierte Winkelsgeschw. $\\theta'(\\tau)$", color="lightgreen", linestyle="--")
    axs2[1].set_title("Harmonisches Pendel (Dimensionslos)")
    axs2[1].set_xlabel("Skalierte Zeit $\\tau = t \\sqrt{g/L}$")
    axs2[1].set_ylabel("Winkel / skalierte Winkelgeschwindigkeit")
    axs2[1].grid(True)
    axs2[1].legend(loc="upper right")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_systems()
