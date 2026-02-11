import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# =============================================================================
# Data
# =============================================================================

# Expected return
mu = np.array([0.01, -0.01, 0.03, -0.03, 0.05, -0.05])
# Covariance matrix
sigma_vals = np.array([
    0.06101, 0.02252, 0.03315, 0.03971, 0.04186, 0.04520,
    0.02252, 0.08762, 0.04137, 0.04728, 0.05241, 0.05310,
    0.03315, 0.04137, 0.10562, 0.06210, 0.06885, 0.06574,
    0.03971, 0.04728, 0.06210, 0.11357, 0.07801, 0.07790,
    0.04186, 0.05241, 0.06885, 0.07801, 0.19892, 0.09424,
    0.04520, 0.05310, 0.06574, 0.07790, 0.09424, 0.36240
])
sigma = sigma_vals.reshape((6, 6))  # 6x6 covariance matrix



# =============================================================================
# Part I : Equal risk contribution
# =============================================================================
# Objective function and gradient, written separately

# Question 1 - What is the objective function?
def obj_logw_long_only(w):
    obj = #...#
    return obj

# Question 2 - What is the associated gradient?
def grad_logw_long_only(w):
    grad = #...#
    return grad

# Constraint: portfolio risk (volatility) equals or is below target.
# We set it as inequality: riskTarget - sqrt(w^T Σ w) >= 0.
def constr_volatility(w, riskTarget, covMatrix):
    # Question 3 - What is the volatility of the portfolio?
    risk = #...#
    
    # Constraint function: riskTarget - risk must be >= 0.
    con_value = riskTarget - risk
    
    # Question 4 - What is the associated gradient?
    jac = #...#
    
    return con_value, jac

def constr_fun(w, riskTarget, covMatrix):
    con_val, _ = constr_volatility(w, riskTarget, covMatrix)
    return con_val

def constr_jac(w, riskTarget, covMatrix):
    jac = constr_volatility(w, riskTarget, covMatrix)
    return jac


# Optimisation code
numAssets = sigma.shape[1]
initial = np.ones(numAssets)
initial = initial / np.sum(initial)
riskTarget = np.sqrt(np.dot(initial, np.dot(sigma, initial)))

# Define bounds: each weight between 1e-6 and 1
bounds = [(1e-6, 1) for _ in range(numAssets)]
# Notes:
# Lower bound is used at 1e-6 and not 0 to ensure that weights remain > 0 for log function.


# Run optimization (long-only)
cons = {'type': 'ineq',
        'fun': lambda w: constr_fun(w, riskTarget, sigma),
        'jac': lambda w: constr_jac(w, riskTarget, sigma)}

res0 = minimize(fun = #...#,
                x0 = initial,
                method = 'SLSQP',
                jac = #...#,
                bounds = bounds,
                constraints = cons,
                options = {'maxiter': 500, 'disp': False})


print("Part I : Equal risk contribution")
# Question 5 & 6 - What is the optimized portfolio? What are the risk contributions of the assets? Check that the risk contributions of the assets are identical

# Normalize solution to sum to one
w_sol = #...#
print("Optimal weights:", w_sol)
# Compute risk contributions of each asset:
risk_contrib = #...#
print("Risk contributions:", risk_contrib)
# Compute total risk of portfolio:
total_risk = #...#
print("Portfolio risk:", total_risk)



# =============================================================================
# Part II : Long / short optimisation
# =============================================================================

# Objective: minimize - sum( |mu| * log(|w|) )
def obj_logw_long_short(w, mu):
    obj = -np.sum(np.abs(mu) * np.log(np.abs(w)))
    return obj

def grad_logw_long_short(w, mu):
    grad = -np.abs(mu) / w
    return grad

# Use same volatility constraint as before (using riskTarget computed from initial weights)
# Set initial weights: +1 for mu>0, -1 for mu<0, normalized by count of positives
initial_ls = np.zeros(numAssets)
initial_ls[mu > 0] = 1
initial_ls[mu < 0] = -1
count_positive = np.sum(mu > 0)
initial_ls = initial_ls / count_positive
riskTarget_ls = np.sqrt(np.dot(initial_ls, np.dot(sigma, initial_ls)))

#Question 7 & 8  - Determine lb and ub
# Set bounds: for assets with mu > 0: [1e-6, 1], for mu < 0: [-1, -1e-6]
bounds_ls = #...#

# Constraint remains the same form (volatility constraint)
cons_ls = {'type': 'ineq',
           'fun': lambda w: constr_fun(w, riskTarget_ls, sigma),
           'jac': lambda w: constr_jac(w, riskTarget_ls, sigma)}

res1 = minimize(fun = #...#,
                x0 = initial_ls,
                args = (mu),
                method = 'SLSQP',
                jac = #...#,
                bounds = bounds_ls,
                constraints = cons_ls,
                options = {'maxiter': 500, 'disp': False})

print("Part II : Long / short optimisation")
# Question 9 - What is the portfolio resulting from this optimization?
#...#


# =============================================================================
# Part III : Long / short optimisation with market neutrality constraint
# =============================================================================

# Define equality constraint: sum(w) == 0
def eq_market_neutral(w):
    return np.sum(w)

def eq_market_neutral_jac(w):
    return np.ones_like(w)

cons_eq = [{'type': 'ineq',
            'fun': lambda w: constr_fun(w, riskTarget_ls, sigma),
            'jac': lambda w: constr_jac(w, riskTarget_ls, sigma)},
           {'type': 'eq',
            'fun': lambda w: eq_market_neutral(w),
            'jac': lambda w: eq_market_neutral_jac(w)}]

res2 = minimize(fun = #...#,
                x0 = initial_ls,
                args = (mu),
                method = 'SLSQP',
                jac = #...#,
                bounds = bounds_ls,
                constraints = cons_eq,
                options = {'maxiter': 500, 'disp': False})

print("Part III : Long / short optimisation with market neutrality constraint")
# Question 10 - Compare the result with other optimizations
#...#


# =============================================================================
# Part IV : Rolling ERC Portfolios
# =============================================================================

# Question 11 - Building ERC rolling portfolios

# Read CSV data (adjust the path if needed)
# The CSV is assumed to use ";" as separator, with decimal ".", and the first column as index.
stocks = pd.read_csv("TD2_MVO.csv", sep = ";", decimal = ".", index_col = 0)
# Assume there is a column "Index" for dates (if not, you can use the index)
dates = stocks.index
# Select the four stock columns
values = stocks[['Walmart', 'Nike', 'CocaCola', 'Citigroup']]
print("Data shape (prices):", values.shape)


# Compute returns for each asset 
returns = #...#

# Annualize the covariance matrix (assuming 252 trading days)
Sigma_ret = #...#

numAssets_mvo = Sigma_ret.shape[0]
num_obs = returns.shape[0]


index_rebal = np.arange(500, num_obs - 22 + 1, 22)

# Prepare an empty weights array (rows = observations, columns = assets)
Weights = np.full((num_obs, numAssets_mvo), np.nan)

# Use equal weights as initial guess
initial_mvo = np.ones(numAssets_mvo) / numAssets_mvo

# Loop over rebalancing dates; use the prior 252 observations for covariance estimation.
for Ind in index_rebal:
    start_idx = Ind - 252
    end_idx = Ind + 1  # include Ind (since Python slices are exclusive at the end)
    if start_idx < 0:
        continue
    returns_tmp = returns.iloc[start_idx:end_idx, :]
    Sigma_tmp = returns_tmp.cov() * 252
    riskTarget_tmp = np.sqrt(np.dot(initial_mvo, np.dot(Sigma_tmp, initial_mvo)))
    
    # Set up optimizer for long-only ERC (using our earlier long-only objective)
    cons_tmp = #...#
    
    bounds = #...#
    
    res_tmp = minimize(#...#)
    
    rpWeight = #...#
    
    # Fill the weights for the next 22 observations (from Ind+1 to Ind+22)
    end_fill = Ind + 22
    if end_fill > num_obs:
        end_fill = num_obs
        
    Weights[Ind+1:end_fill, :] = np.tile(rpWeight, (end_fill - (Ind+1), 1))

# Remove any rows that are all NaN (i.e. before the first rebalancing)
valid_weights = Weights[~np.isnan(Weights).all(axis=1)]
time_axis = np.arange(valid_weights.shape[0])

# Create a DataFrame for plotting
df_weights = pd.DataFrame(valid_weights, columns = ['Walmart', 'Nike', 'CocaCola', 'Citigroup'])
df_weights.index = time_axis

# Question 12 - Draw a graph of these portfolios
# Plot stacked area chart of allocations over time
#...#
