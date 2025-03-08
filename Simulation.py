import numpy as np
import matplotlib.pyplot as plt

class MonteCarloSimulator_BlackScholes:
    def __init__(self, S0, r, sigma, T, k, n_simulations):
        self.S0 = S0  # Prix initial de l'actif
        self.r = r  # Taux sans risque
        self.sigma = sigma  # Volatilité
        self.T = T  # Maturité
        self.k = k  # Prix d'exercice (strike)
        self.n_simulations = n_simulations  # Nombre de simulations
    def Call_Price_BlackScholes(self):
        Wt = np.random.normal(0, np.sqrt(self.T), self.n_simulations)  
        St = self.S0 * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * Wt)  
        payoffs = np.maximum(St - self.k, 0)  
        CallPrice = np.exp(-self.r * self.T) * np.mean(payoffs)  
        return CallPrice
    def Put_Price_BlackScholes(self):
        Wt = np.random.normal(0, np.sqrt(self.T), self.n_simulations)
        St = self.S0 * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * Wt)
        payoffs = np.maximum(self.k - St, 0)  
        PutPrice = np.exp(-self.r * self.T) * np.mean(payoffs)  
        return PutPrice
   
    

class MonteCarloSimulator_MertonJumpDiffusion:
    def __init__(self, S0, r, sigma, T, k, n_simulations, lambda_jump, mu_jump, sigma_jump, n_pas):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.k = k
        self.n_simulations = n_simulations
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
        self.n_pas = n_pas
    def Simulate_Jump_Diffusion(self):
        dt = self.T / self.n_pas  # Intervalle de temps
        St = np.zeros((self.n_simulations, self.n_pas + 1))  # Matrice pour stocker les trajectoires
        St[:, 0] = self.S0  # Initialiser le prix initial
        for i in range(self.n_simulations):
            # Simulation du mouvement brownien et des sauts
            Wt = np.random.normal(0, np.sqrt(dt), self.n_pas)  # Mouvement brownien
            N = np.random.poisson(self.lambda_jump * dt, self.n_pas)  # Nombre de sauts
            jumps = np.random.normal(self.mu_jump, self.sigma_jump, self.n_pas)  # Amplitude des sauts
            # Simuler le prix de l'actif avec sauts
            for t in range(1, self.n_pas + 1):
                St[i, t] = St[i, t - 1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * Wt[t - 1] + jumps[t - 1] * N[t - 1])
        return St
    def Call_Price_MertonJumpDiffusion(self):
        St = self.Simulate_Jump_Diffusion()  # Simuler les trajectoires avec sauts
        payoffs = np.maximum(St[:, -1] - self.k, 0)  # Calcul des payoffs de l'option Call
        CallPrice = np.exp(-self.r * self.T) * np.mean(payoffs)  # Actualisation du prix de l'option Call
        return CallPrice
    def Put_Price_MertonJumpDiffusion(self):
        St = self.Simulate_Jump_Diffusion()  # Simuler les trajectoires avec sauts
        payoffs = np.maximum(self.k - St[:, -1], 0)  # Calcul des payoffs de l'option Put
        PutPrice = np.exp(-self.r * self.T) * np.mean(payoffs)  # Actualisation du prix de l'option Put
        return PutPrice
    
    

    


