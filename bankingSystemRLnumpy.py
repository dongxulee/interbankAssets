import mesa
import numpy as np
import pandas as pd
from eisenbergNoe import eisenbergNoe
from collections import defaultdict

class Bank(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # assets
        self.portfolio = 0.        # initialize when creating the bank, change in borrowing and lending
        self.lending = 0.         
        # liabilities
        self.borrowing = 0.       
        # equity
        self.equity = 0.           # initialize when creating the bank, update in updateBlanceSheet()
        # leverage ratio
        self.leverage = 0.        
        # if a bank is solvent
        self.default = False      # change at clearingDebt()
        # accumulated action probabilities in an episode  
        self.log_probs = 0.
    
    def updateBlanceSheet(self):
        self.equity = self.portfolio + self.lending - self.borrowing
        self.leverage = (self.portfolio + self.lending) / self.equity
        
    def borrowRequest(self):
        for _ in range(self.model.num_borrowing):
            if self.leverage < self.model.leverageRatio:
                # randomly choose a bank to borrow from the trust matrix
                target = np.random.choice(self.model.N, p=self.model.trustMatrix[self.unique_id])
                # choose a borrowing amount equal to the equity capital
                amount = self.equity * self.model.sizeOfBorrowing
                # bring out the target bank and let him decide whether to lend
                other_agent = self.model.schedule.agents[target]
                # if the lending decision is made, update the balance sheet
                if other_agent.portfolio * (1-self.model.capitalReserve) > amount:
                    # lending ratio is based on the policy function
                    state = np.array([other_agent.lending/other_agent.portfolio, other_agent.borrowing/other_agent.portfolio, amount/other_agent.portfolio,
                                            self.lending/self.portfolio, self.borrowing/self.portfolio, self.model.sizeOfBorrowing])
                    # sampled action
                    mean = self.model.policy(state)
                    a = mean + np.random.normal(0, 0.01)
                    # log_probs collection
                    self.log_probs += (a-mean)*state
                    if np.random.rand() < 1.0/(1.0+np.exp(-a)):
                        # update borrowers balance sheet
                        self.model.L[self.unique_id, target] += amount 
                        self.portfolio += amount
                        self.model.e[self.unique_id] = self.portfolio
                        self.borrowing += amount
                        self.updateBlanceSheet()
                        self.model.concentrationParameter[self.unique_id, target] += 1.
                        # update lenders balance sheet 
                        other_agent.portfolio -= amount
                        self.model.e[other_agent.unique_id] = other_agent.portfolio
                        other_agent.lending += amount
                        # asset and equity amount remain unchanged, leverage ratio also remains unchanged
                
    def returnOnPortfolio(self):
        self.portfolio = self.portfolio * (1+self.model.portfolioReturnRate)
        self.model.e[self.unique_id] = self.portfolio
        self.updateBlanceSheet()
    
    def reset(self):
        self.portfolio = self.model.e[self.unique_id][0]
        self.lending = 0.    
        # liabilities
        self.borrowing = 0.      
        # equity
        self.equity = self.portfolio   
        # leverage ratio
        self.leverage = 1.0
        if self.portfolio == 0.:
            self.default = True
    
    def step(self):
        if not self.default:
            self.borrowRequest()
            self.returnOnPortfolio()
    

class bankingSystem(mesa.Model):
    def __init__(self, banksFile, 
                 leverageRatio, 
                 capitalReserve,
                 num_borrowing,
                 sizeOfBorrowing,
                 num_banks, 
                 alpha = 0.99,
                 beta = 0.99,
                 concentrationParameter = None, 
                 fedRate = 0., 
                 portfolioReturnRate = 0., 
                 liquidityShockNum = 0,
                 shockSize = 0.0,
                 shockDuration=[-1,-1],
                 policy = None):
        
        # interest rate
        self.fedRate = fedRate
        # portfolio return rate
        self.portfolioReturnRate = portfolioReturnRate
        # number of liquidity shocks
        self.liquidityShockNum = liquidityShockNum 
        # size of the shock
        self.shockSize = shockSize
        # time of the shock
        self.shockDuration = shockDuration
        # shocked banks
        self.shockedBanks = defaultdict(int)
        # asset recovery rate 
        self.alpha = alpha
        # interbank loan recovery rate
        self.beta = beta
        
        # read in banks equity capital
        banksData = pd.read_csv(banksFile).iloc[:num_banks,:]
        self.banks = banksData["bank"]
        self.N = num_banks
        self.leverageRatio = leverageRatio
        self.capitalReserve = capitalReserve
        self.num_borrowing = num_borrowing
        # RL realted parameters
        self.policy = policy 
        self.sizeOfBorrowing = sizeOfBorrowing
        # start with a uniform distribution of trust, using Dirichlet distribution as a conjugate prior
        # we also introduce a time decay factor for trust       
        if concentrationParameter is None:
            self.concentrationParameter = np.ones((self.N,self.N))
            np.fill_diagonal(self.concentrationParameter, 0.)
        else:
            self.concentrationParameter = concentrationParameter
        self.trustMatrix = self.concentrationParameter / self.concentrationParameter.sum(axis=1, keepdims=True)
        # liability matrix 
        self.L = np.zeros((self.N,self.N))
        # asset matrix
        self.e = banksData["equity"].values.reshape(self.N,1)
        # create a schedule for banks
        self.schedule = mesa.time.RandomActivation(self)
    
        # create banks and put them in schedule
        for i in range(self.N):
            a = Bank(i, self)
            a.portfolio = banksData["equity"][i]
            a.equity = banksData["equity"][i]
            self.schedule.add(a)
            
        self.datacollector = mesa.DataCollector(
            model_reporters={ "Trust Matrix": "trustMatrix", 
                             "Liability Matrix": "L",
                             "Asset Matrix": "e"}
            # ,
            # agent_reporters={"PortfolioValue": "portfolio",
            #                     "Lending": "lending",
            #                     "Borrowing": "borrowing", 
            #                     "Equity": "equity",
            #                     "Default": "default",
            #                     "Leverage": "leverage"}
            )
        
    def updateTrustMatrix(self):
        # add time decay of concentration parameter
        self.concentrationParameter = self.concentrationParameter / self.concentrationParameter.sum(axis=1, keepdims=True) * (self.N - 1)
        self.trustMatrix = self.concentrationParameter / (self.N - 1)
            
    def clearingDebt(self):
        # Returns the new portfolio value after clearing debt
        _, e, insolventBanks = eisenbergNoe(self.L*(1+self.fedRate), self.e, self.alpha, self.beta)
        self.e = e
        # reset the Liabilities matrix after clearing debt
        self.L = np.zeros((self.N,self.N))
        if len(insolventBanks) > 0:
            self.concentrationParameter[:,insolventBanks] = 0
        for agent in self.schedule.agents:
            agent.reset()
            
    def liquidityShock(self):
        # liquidity shock to banks portfolio
        if self.schedule.time >= self.shockDuration[0] and self.schedule.time <= self.shockDuration[1]:
            if self.liquidityShockNum > 0:
                for _ in range(self.liquidityShockNum):
                    # randomly choose a bank to be insolvent
                    exposedBank = np.random.choice(self.N)
                    # set the bank's equity to drop
                    self.e[exposedBank] *= (1-self.shockSize)
                    self.shockedBanks[exposedBank] += 1
    def simulate(self):
        self.schedule.step()
        self.liquidityShock()
        self.updateTrustMatrix()
        # self.datacollector.collect(self)
        self.clearingDebt()
