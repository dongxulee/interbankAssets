from bankingSystemLearning import * 
from helperFunctions import *

modele = pd.read_csv("balanceSheetAnalysis/banksData_2022.csv")[:100]["assets"].values.reshape(-1,1)
def R_tau(model):
    value = (model.e - modele)/modele
    return value

theta = np.zeros((2,1))
for i in range(1000):
    # simulation and data collection
    simulationSteps = 500
    model = bankingSystem(banksFile="balanceSheetAnalysis/banksData_2022.csv", # csv file used to initialize the bank agents
                    leverageRatio = 10.0,                                     # leverage ratio upper bound for all banks
                    depositReserve = 0.20,                                     # capital reserve as a ratio of portfolio value
                    num_borrowing= 20,                                        # number of borrowing request per bank per step
                    sizeOfBorrowing = 1.0/3,                                      # size of borrowing as a ratio of equity capital
                    num_banks=100,                                            # number of banks in the system 
                    alpha = 0.5,                                              # portfolio recovery rate                           
                    beta = 0.9,                                               # interbank loan recovery rate
                    fedRate = 0.04,                                            # interest rate on borrowing   
                    portfolioReturnRate = 0.10,                                # return rate on portfolio
                    returnVolatiliy = 0.18,
                    gammas = np.linspace(4.0,0.5,100).reshape(-1,1),
                    returnCorrelation = np.diag(np.ones(100)),
                    liquidityShockNum = 1,                                    # number of liquidity shocks per step      
                    shockSize = 0.,                                          # size of the shock
                    shockDuration =  [300, 305],
                    theta=theta) # duration of the shock
                    
    model.datacollector.collect(model)
    for _ in range(simulationSteps):
        model.simulate()
    
    reward = R_tau(model)
    theta += (model.grad * reward).mean(axis = 0).reshape(-1,1) * 0.01
    print(i)
    print(theta)
    print(reward.sum())