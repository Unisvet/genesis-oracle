# Genesis Oracle 🌌

Genesis Oracle is a Python-based simulation engine designed to explore and visualize physical systems through both standard and dimensionless differential equations.

## Features 🚀

- **Radioactive Decay Simulation**: Compare physical decay (atoms/second) with dimensionless scaled time.
- **Harmonic Pendulum**: Visualize angular displacement and velocity using SI units vs. scaled dimensionless units ($\tau = t \sqrt{g/L}$).
- **Numerical Integration**: Powered by `scipy.integrate.solve_ivp` for accurate results.
- **Visualization**: Clear, comparative plots using `matplotlib`.

## Installation 🛠️

Ensure you have Python 3.13+ installed.

```bash
# Install dependencies
pip install numpy matplotlib scipy
```

## Usage 📈

Run the simulation script to generate comparison plots:

```bash
python src/main.py
```

## Project Structure 📂

- `src/main.py`: Core simulation logic and plotting routines.
- `main.py`: Entry point for the package.
- `agents/`: AI agents for system analysis (planned).
- `docs/`: Technical documentation.
- `data/`: Simulation output data.

## Physics Overview 🧪

### Radioactive Decay
Standard: $\frac{dN}{dt} = -\lambda N$
Dimensionless: $\frac{dN'}{d\tau} = -N'$ where $N' = \frac{N}{N_0}$ and $\tau = \lambda t$.

### Pendulum
Standard: $\frac{d^2\theta}{dt^2} + \frac{g}{L}\sin\theta = 0$
Dimensionless: $\frac{d^2\theta}{d\tau^2} + \sin\theta = 0$ where $\tau = t \sqrt{\frac{g}{L}}$.

---
Developed as part of the Genesis Oracle initiative.
