from bankingSystem import * 
from helperFunctions import *

modele = pd.read_csv("balanceSheetAnalysis/banksData_2022.csv")[:100][["equity", "deposit"]].sum(axis=1).values.reshape(-1,1)
def R_tau(model):
    value = (model.e - modele)/modele
    return value.flatten()

def f(gammas):
    gammas = gammas.reshape(-1,1)
    # simulation and data collection
    simulationSteps = 500
    model = bankingSystem(banksFile="balanceSheetAnalysis/banksData_2022.csv", # csv file used to initialize the bank agents
                    leverageRatio = 10.0,                                     # leverage ratio upper bound for all banks
                    capitalReserve = 0.2,                                     # capital reserve as a ratio of deposits
                    num_borrowing= 20,                                        # number of borrowing request per bank per step
                    sizeOfBorrowing = 1.0/3.0,                                      # size of borrowing as a ratio of equity capital
                    num_banks=100,                                            # number of banks in the system 
                    alpha = 0.99,                                              # portfolio recovery rate                           
                    beta = 0.99,                                               # interbank loan recovery rate
                    fedRate = 0.04,                                            # interest rate on borrowing   
                    portfolioReturnRate = 0.10,                                # return rate on portfolio
                    returnVolatiliy = 0.18,
                    returnCorrelation = np.diag(np.ones(100)),
                    gammas = gammas,
                    liquidityShockNum = 2,                                    # number of liquidity shocks per step      
                    shockSize = 0.10,                                          # size of the shock
                    shockDuration =  [simulationSteps // 10 * 6, simulationSteps // 10 * 7]) # duration of the shock

    model.datacollector.collect(model)
    for i in range(simulationSteps):
        model.simulate()
    
    return R_tau(model).sum()


if __name__ == "__main__":
    import nevergrad as ng
    # Define the bounds for x
    x_init = np.random.uniform(low=0.5, high=3, size=(100,))
    instrum = ng.p.Instrumentation(
        ng.p.Array(init=x_init).set_bounds(0.5, 4)
    )

    # Optimize the function
    optimizer = ng.optimizers.NGOpt(instrum, budget=1000)
    recommendation = optimizer.minimize(f, verbosity = True)
    print("x value: ", recommendation.value[0][0])
    print("f value: ", f(recommendation.value[0][0]))