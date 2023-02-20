from bankingSystemLearning import * 
from helperFunctions import *

con = np.load("concentrationParams.npy")
modele = pd.read_csv("balanceSheetAnalysis/banksData_2022.csv")[:100]["assets"].values.reshape(-1,1)
cMatrix = np.ones((100,100))*0.8
np.fill_diagonal(cMatrix, 1)

def R_tau(model):
    value = (model.e - modele)/modele
    return value.sum()

def f(theta):
    theta = np.array(theta).reshape(-1,1) 
    # simulation and data collection
    simulationSteps = 500
    cMatrix = np.ones((100,100))*0.8
    np.fill_diagonal(cMatrix, 1)
    model = bankingSystem(banksFile="balanceSheetAnalysis/banksData_2022.csv", # csv file used to initialize the bank agents
                    leverageRatio = 20.0,                                     # leverage ratio upper bound for all banks
                    depositReserve = 0.2,                                     # capital reserve as a ratio of portfolio value
                    num_borrowing= 20,                                        # number of borrowing request per bank per step
                    sizeOfBorrowing = 1.0, concentrationParameter = con,                                      # size of borrowing as a ratio of equity capital
                    num_banks=100,                                            # number of banks in the system 
                    alpha = 0.5,                                              # portfolio recovery rate                           
                    beta = 0.9,                                               # interbank loan recovery rate
                    fedRate = 0.04,                                            # interest rate on borrowing   
                    portfolioReturnRate = 0.10,          
                    # return rate on portfolio
                    returnVolatiliy = 0.18,
                    returnCorrelation = cMatrix,
                    liquidityShockNum = 10,                                    # number of liquidity shocks per step      
                    shockSize = 0.02,                                          # size of the shock
                    shockDuration = [300, 302]) # duration of the shock
                    
    for _ in range(simulationSteps):
        model.simulate()
    
    reward = R_tau(model)
    return [-reward]

import pyswarms as ps
from sklearn.neighbors import KNeighborsRegressor

# Call the PSO optimization
max_bound = [3, 3]
min_bound = [-3, -3]
bounds = (min_bound, max_bound)
# Initialize swarm
options = {'c1': 0.5, 'c2': 0.2, 'w':0.9}

optimizer = ps.single.GlobalBestPSO(n_particles=40, dimensions=2, options=options, bounds=bounds)

# The optimal solution is stored in xopt and fopt
cost, pos = optimizer.optimize(f, iters=500, n_processes = 40, verbose=True)

X = np.vstack(optimizer.pos_history)
y = np.hstack(optimizer.costs)
np.save("samplePoints1.npy", X)
np.save("sampleValues1.npy", y)
print("Best Cost", cost)
print("Best Position", pos)

import seaborn as sns
X = np.load("samplePoints1.npy")
y = np.load("sampleValues1.npy")

plt.rcParams.update({'font.size': 20})
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
fig.suptitle('Search in Parameter Space')
sns.kdeplot(X[:,0], X[:,1], x =r'$\theta_1$', y= r'$\theta_2$', shade=True, ax = axs[0])
axs[0].set_xlabel(r'$\theta_1$')
axs[0].set_ylabel(r'$\theta_2$')
neigh = KNeighborsRegressor(n_neighbors=500)
neigh.fit(X,y)
xx = np.linspace(-3, 3, num=100)
yy = np.linspace(-3, 3, num=100)
XX, YY = np.meshgrid(xx, yy)
X_flat = XX.ravel()[:, np.newaxis]
Y_flat = YY.ravel()[:, np.newaxis]
X_grid = np.concatenate((X_flat, Y_flat), axis=1)
y_pred = neigh.predict(X_grid)
Z = y_pred.reshape(XX.shape)
contour = axs[1].contourf(XX, YY, Z, cmap='viridis')
axs[1].scatter(pos[0], pos[1] + 0.5, marker='*', s=200, c='r')
axs[1].set_xlabel(r'$\theta_1$')
axs[1].set_ylabel(r'$\theta_2$')
fig.colorbar(contour, ax=axs[1])