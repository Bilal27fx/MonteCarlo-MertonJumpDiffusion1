import numpy as np
import plotly.graph_objects as go
import streamlit as st
from Simulation import MonteCarloSimulator_BlackScholes, MonteCarloSimulator_MertonJumpDiffusion

# Streamlit interface configuration
st.title("Monte Carlo Simulation for Options")

st.sidebar.header("Black-Scholes Option Parameters")

# Inputs for the Black-Scholes model without the parameter names in parentheses
S0 = st.sidebar.slider("Initial Price", 50, 200, 100)
r = st.sidebar.slider("Risk-Free Rate", 0.0, 0.1, 0.05)
sigma = st.sidebar.slider("Volatility", 0.0, 1.0, 0.2)
T = st.sidebar.slider("Maturity", 0.1, 10.0, 1.0)
k = st.sidebar.slider("Strike Price", 50, 200, 100)
n_simulations = st.sidebar.slider("Number of Simulations", 1000, 10000, 10000)

st.sidebar.header("Merton Jump Diffusion Option Parameters")

# Inputs for the Merton Jump Diffusion model without the parameter names in parentheses
lambda_jump = st.sidebar.slider("Jump Intensity", 0.0, 1.0, 0.2)
mu_jump = st.sidebar.slider("Jump Mean", -1.0, 0.0, -0.15)
sigma_jump = st.sidebar.slider("Jump Volatility", 0.0, 1.0, 0.1)
n_steps = st.sidebar.slider("Number of Time Steps", 100, 500, 100)

# Create the simulators
bs_simulator = MonteCarloSimulator_BlackScholes(S0, r, sigma, T, k, n_simulations)
mj_simulator = MonteCarloSimulator_MertonJumpDiffusion(S0, r, sigma, T, k, n_simulations, lambda_jump, mu_jump, sigma_jump, n_steps)

# Calculate option prices
call_price_bs = bs_simulator.Call_Price_BlackScholes()
put_price_bs = bs_simulator.Put_Price_BlackScholes()
call_price_mj = mj_simulator.Call_Price_MertonJumpDiffusion()
put_price_mj = mj_simulator.Put_Price_MertonJumpDiffusion()

# Display the results with colored squares
st.subheader("                    Black-Scholes Option Price")

# Display the results with centered squares for Call and Put prices (Black-Scholes)
st.markdown(f"""
    <div style="display: flex; justify-content: center; gap: 30px; margin-top: 20px;">
        <div style="width: 180px; height: 100px; background-color: #4CAF50; display: flex; align-items: center; justify-content: center; color: white; font-size: 18px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            Call: {call_price_bs:.2f}
        </div>
        <div style="width: 180px; height: 100px; background-color: #f44336; display: flex; align-items: center; justify-content: center; color: white; font-size: 18px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            Put: {put_price_bs:.2f}
        </div>
    </div>
""", unsafe_allow_html=True)

st.subheader("                 Merton Jump Diffusion Option Price")

# Display with centered squares for Call and Put prices (Merton Jump Diffusion)
st.markdown(f"""
    <div style="display: flex; justify-content: center; gap: 30px; margin-top: 20px;">
        <div style="width: 180px; height: 100px; background-color: #2196F3; display: flex; align-items: center; justify-content: center; color: white; font-size: 18px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            Call: {call_price_mj:.2f}
        </div>
        <div style="width: 180px; height: 100px; background-color: #FFC107; display: flex; align-items: center; justify-content: center; color: white; font-size: 18px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            Put: {put_price_mj:.2f}
        </div>
    </div>
""", unsafe_allow_html=True)

# 3D Plot with Plotly for option prices
st.subheader("Interactive 3D Option Price Visualization")

# Function to compute option prices dynamically for 3D plot
def calculate_option_prices(S0, r, sigma, T, k_values, n_simulations):
    prices = np.zeros((len(k_values), len(T)))
    for i, T_val in enumerate(T):
        for j, k_val in enumerate(k_values):
            bs_simulator = MonteCarloSimulator_BlackScholes(S0, r, sigma, T_val, k_val, n_simulations)
            call_price = bs_simulator.Call_Price_BlackScholes()
            prices[j, i] = call_price
    return prices

# Value ranges for K (Strike Price) and T (Time to Maturity)
k_values = np.linspace(50, 200, 30)  # Prices of exercise from 50 to 200
T_values = np.linspace(0.1, 5.0, 30)  # Time to maturity from 0.1 to 5 years

# Calculate option prices
option_prices = calculate_option_prices(S0, r, sigma, T_values, k_values, n_simulations)

# Create an interactive 3D plot with Plotly
fig = go.Figure(data=[go.Surface(
    z=option_prices,
    x=k_values,  # x is Strike Price (K)
    y=T_values,  # y is Time to Maturity (T)
    colorscale='Jet',  # Heatmap color scale
    colorbar=dict(title='Option Price'),
)])

# Customize the layout
fig.update_layout(
    title="3D Option Price Surface (Black-Scholes)",
    scene=dict(
        xaxis_title='Strike Price (K)',
        yaxis_title='Time to Maturity (T)',
        zaxis_title='Option Price (Call)'
    ),
    height=800
)

# Display the Plotly plot in Streamlit
st.plotly_chart(fig)
