import numpy as np
import matplotlib.pyplot as plt
import Simulation
import streamlit as st

# Configuration de l'interface Streamlit
st.title("Simulation de Monte Carlo pour Options")

st.sidebar.header("Paramètres d'options (Black-Scholes)")

# Inputs pour le modèle Black-Scholes
S0 = st.sidebar.slider("Prix initial (S0)", 50, 200, 100)
r = st.sidebar.slider("Taux sans risque (r)", 0.0, 0.1, 0.05)
sigma = st.sidebar.slider("Volatilité (sigma)", 0.0, 1.0, 0.2)
T = st.sidebar.slider("Maturité (T en années)", 0.1, 10.0, 1.0)
k = st.sidebar.slider("Prix d'exercice (k)", 50, 200, 100)
n_simulations = st.sidebar.slider("Nombre de simulations", 1000, 10000, 10000)

st.sidebar.header("Paramètres d'options (Merton Jump Diffusion)")

# Inputs pour le modèle Merton Jump Diffusion
lambda_jump = st.sidebar.slider("Intensité des sauts (lambda)", 0.0, 1.0, 0.2)
mu_jump = st.sidebar.slider("Moyenne des sauts (mu)", -1.0, 0.0, -0.15)
sigma_jump = st.sidebar.slider("Volatilité des sauts (sigma_jump)", 0.0, 1.0, 0.1)
n_pas = st.sidebar.slider("Nombre de pas de temps (n_pas)", 100, 500, 100)

# Créer les simulateurs
bs_simulator = MonteCarloSimulator_BlackScholes(S0, r, sigma, T, k, n_simulations)
mj_simulator = MonteCarloSimulator_MertonJumpDiffusion(S0, r, sigma, T, k, n_simulations, lambda_jump, mu_jump, sigma_jump, n_pas)

# Calcul des prix des options
call_price_bs = bs_simulator.Call_Price_BlackScholes()
put_price_bs = bs_simulator.Put_Price_BlackScholes()
call_price_mj = mj_simulator.Call_Price_MertonJumpDiffusion()
put_price_mj = mj_simulator.Put_Price_MertonJumpDiffusion()

# Affichage des résultats
st.subheader("Prix de l'option (Black-Scholes)")
st.write(f"Prix de l'option Call: {call_price_bs:.2f}")
st.write(f"Prix de l'option Put: {put_price_bs:.2f}")

st.subheader("Prix de l'option (Merton Jump Diffusion)")
st.write(f"Prix de l'option Call: {call_price_mj:.2f}")
st.write(f"Prix de l'option Put: {put_price_mj:.2f}")

# Option de visualisation des trajectoires simulées
st.subheader("Graphique des Trajectoires Simulées (Merton Jump Diffusion)")

St_mj = mj_simulator.Simulate_Jump_Diffusion()

# Tracer les trajectoires simulées
fig, ax = plt.subplots()
for i in range(min(10, n_simulations)):  # Affiche les 10 premières trajectoires
    ax.plot(np.linspace(0, T, n_pas + 1), St_mj[i, :], lw=0.8)

ax.set_title("Trajectoires Simulées (Merton Jump Diffusion)")
ax.set_xlabel("Temps (années)")
ax.set_ylabel("Prix de l'actif")
st.pyplot(fig)






