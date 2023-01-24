from bankingSystem import * 
from helperFunctions import *

modele = pd.read_csv("balanceSheetAnalysis/banksData_2022.csv")[:100]["assets"].values.reshape(-1,1)
def R_tau(model):
    value = (model.e - modele)/modele
    return value.flatten().sum()

def f(gammas):
    gammas = gammas.reshape(-1,1)
    # simulation and data collection
    simulationSteps = 500
    model = bankingSystem(banksFile="balanceSheetAnalysis/banksData_2022.csv", # csv file used to initialize the bank agents
                    leverageRatio = 10.0,                                     # leverage ratio upper bound for all banks
                    depositReserve = 0.80,                                     # capital reserve as a ratio of portfolio value
                    num_borrowing= 20,                                        # number of borrowing request per bank per step
                    sizeOfBorrowing = 1.0/3,                                      # size of borrowing as a ratio of equity capital
                    num_banks=100,                                            # number of banks in the system 
                    alpha = 0.5,                                              # portfolio recovery rate                           
                    beta = 0.9,                                               # interbank loan recovery rate
                    fedRate = 0.04,                                            # interest rate on borrowing   
                    portfolioReturnRate = 0.10,                                # return rate on portfolio
                    returnVolatiliy = 0.18,
                    gammas = gammas,
                    returnCorrelation = np.diag(np.ones(100)),
                    liquidityShockNum = 1,                                    # number of liquidity shocks per step      
                    shockSize = 0.,                                          # size of the shock
                    shockDuration =  [300, 305]) # duration of the shock
                    
    model.datacollector.collect(model)
    for _ in range(simulationSteps):
        model.simulate()
    
    return -R_tau(model)


if __name__ == "__main__":
    import nevergrad as ng
    from concurrent import futures
    # Define the bounds for x
    x_init = np.random.uniform(low=0.5, high=4, size=(100,))
    instrum = ng.p.Instrumentation(
        ng.p.Array(init=x_init).set_bounds(0.5, 4)
    )

    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=1000, num_workers=10)
    # We use ThreadPoolExecutor for CircleCI but please
    # use the line just below, with ProcessPoolExecutor instead (unless your
    # code is I/O bound rather than CPU bound):
    with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
    #with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
        recommendation = optimizer.minimize(f, executor=executor, batch_mode=True, verbosity = 2)
        print("x value: ", recommendation.value[0][0])
        print("f value: ", f(recommendation.value[0][0]))