from bankingSystemRLnumpy import * 
from helperFunctions import *
from tqdm import tqdm
import warnings 
warnings.filterwarnings('ignore')

class PolicyFunction:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)

    def __call__(self, x): 
        val = np.dot(x, self.weights) 
        return val
    
    def update(self, gradient, stepSize):
        self.weights += stepSize * gradient
    
policy = PolicyFunction(6)

modele = pd.read_csv("balanceSheetAnalysis/banksData_2022.csv")["equity"][:100].values.reshape(-1,1)
def R_tau(model, modele):
    value = (model.e - modele)/modele
    return value.flatten()

Rewards = []
# simulation and data collection
simulationSteps = 500
gradientSteps = 1000

for sim in range(gradientSteps):
    print(sim)
    model = bankingSystem(banksFile="balanceSheetAnalysis/banksData_2022.csv", # csv file used to initialize the bank agents
                        leverageRatio = 11.0,                                     # leverage ratio upper bound for all banks
                        capitalReserve = 0.0,                                     # capital reserve as a ratio of portfolio value
                        num_borrowing= 20,                                        # number of borrowing request per bank per step
                        sizeOfBorrowing = 1,                                      # size of borrowing as a ratio of equity capital
                        num_banks=100,                                            # number of banks in the system 
                        alpha = 0.5,                                              # portfolio recovery rate                           
                        beta = 0.9,                                               # interbank loan recovery rate
                        fedRate = 0.04/252,                                       # interest rate on borrowing   
                        portfolioReturnRate = 0.03/252,                                  # return rate on portfolio
                        liquidityShockNum = 2,                                    # number of liquidity shocks per step      
                        shockSize = 0.1,                                           # size of the shock
                        shockDuration =  [simulationSteps // 10 * 6, simulationSteps // 10 * 7], # duration of the shock
                        policy = policy)                                                    # policy function weights
                        
    #model.datacollector.collect(model)
    for i in tqdm(range(simulationSteps)):
        model.simulate()
    #agent_data = model.datacollector.get_agent_vars_dataframe()
    #model_data = model.datacollector.get_model_vars_dataframe()
    print(policy.weights)
    reward = R_tau(model, modele)
    gradient = np.zeros(6)
    for i in range(reward.size):
        gradient += reward[i] * model.schedule.agents[i].log_probs 
    gradient /= reward.size
    policy.update(gradient, stepSize=0.01)
    Rewards.append(reward)
    np.save("models/weights" + str(sim), policy.weights)
np.save("Rewards", Rewards)
