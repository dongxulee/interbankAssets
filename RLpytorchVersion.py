from bankingSystemRLtorch import * 
from helperFunctions import *
from tqdm import tqdm
import warnings 
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn

# Define the policy function as a PyTorch model
class PolicyFunction(nn.Module):
  def __init__(self, input_size):
    super(PolicyFunction, self).__init__()
    self.fc1 = nn.Linear(input_size, 1)
    self.fc1.weight.data.fill_(0.)
    self.fc1.bias.data.fill_(0.)
    
  
  def forward(self, x):
    x = self.fc1(x)
    x = torch.sigmoid(x)
    return x

policy = PolicyFunction(6)
# mean = policy(torch.tensor([0.0000, 0.0000, 0.0703, 0.0000, 0.0000, 0.0703], dtype=torch.float))
# dist = torch.distributions.Normal(mean, 0.01)
# a = dist.sample()
# like = dist.log_prob(a)
# a.item()

modele = pd.read_csv("balanceSheetAnalysis/banksData_2022.csv")["equity"][:100].values.reshape(-1,1)
def R_tau(model, modele):
    value = (model.e - modele)/modele
    return value.flatten()

Rewards = []
# simulation and data collection
simulationSteps = 500
gradientSteps = 1000
optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

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

    policy.zero_grad()
    reward = R_tau(model, modele)
    loss = 0
    for i in range(len(reward)):
        loss -= reward[i] * model.schedule.agents[i].log_probs
    loss.backward()
    optimizer.step()
    torch.save(policy.state_dict(), 'models/model' + str(sim) + '.pt')
    print(policy.state_dict())
    Rewards.append(reward)
np.save("Rewards", Rewards)