import mesa
import numpy as np
import pandas as pd
from eisenbergNoe import eisenbergNoe

class Bank(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # assets
        self.portfolio = 0.        # initialize when creating the bank, change in borrowing and lending
        self.lending = 0.         
        # liabilities
        self.borrowing = 0.       
        self.deposit = 0.
        # equity
        self.equity = 0.           # initialize when creating the bank, update in updateBlanceSheet()
        # leverage ratio
        self.leverage = 0.        
        # if a bank is solvent
        self.default = False      # change at clearingDebt()
    
    def updateBlanceSheet(self):
        # equity = asset - liability
        self.equity = self.portfolio + self.lending - self.borrowing - self.deposit
        # leverage ratio = asset / equity
        self.leverage = (self.portfolio + self.lending) / self.equity
        
    def borrowRequest(self):
        borrowingAmount = self.model.targetBorrowingLending[self.unique_id]
        if borrowingAmount > 0:
            for _ in range(self.model.num_borrowing):
                if self.leverage < self.model.leverageRatio:
                    # randomly choose a bank to borrow from the trust matrix
                    prob = self.model.trustMatrix[self.unique_id]
                    target = np.random.choice(self.model.N, p=prob)
                    # choose a borrowing amount equal to the equity capital
                    amount = borrowingAmount * self.model.sizeOfBorrowing
                    # bring out the target bank and let him decide whether to lend
                    other_agent = self.model.schedule.agents[target]
                    # if the lending decision is made, update the balance sheet
                    if other_agent.lendDecision(self, amount):
                        self.model.L[self.unique_id, target] += amount
                        self.portfolio += amount
                        self.model.e[self.unique_id] = self.portfolio
                        self.borrowing += amount
                        self.updateBlanceSheet()
                        self.model.concentrationParameter[self.unique_id, target] += 1.
                        borrowingAmount -= amount
                if borrowingAmount <= 0:
                    break
                
    # reinforcement learning update later. 
    def lendDecision(self, borrowingBank, amount):
        lendingAmount = -self.model.targetBorrowingLending[self.unique_id]
        if lendingAmount > amount:
            # collect borrowing banks information, in this version, if the banks have enough liquidity, they will lend 
            # borrowingBank's information could be access through borrowingBank 
            if self.portfolio - self.deposit * self.model.depositReserve > amount and self.leverage < self.model.leverageRatio:
                self.portfolio -= amount 
                self.model.e[self.unique_id] = self.portfolio
                self.lending += amount
                # asset and equity amount remain unchanged, leverage ratio also remains unchanged
                self.model.targetBorrowingLending[self.unique_id] += amount
                return True
            else:
                return False
        else:
            return False
    
    def reset(self):
        self.portfolio = self.model.e[self.unique_id][0]
        # if default 
        if self.portfolio == 0:
            # use portfolio to pay off the deposit 
            self.deposit = 0.
            self.lending = 0.
            self.borrowing = 0.
            self.leverage = 0.
            self.equity = 0.
            self.default = True 
        else:
            self.lending = 0.    
            self.borrowing = 0.      
            self.updateBlanceSheet()
            # if the leverage ratio is too high, the bank will pay off the deposit, equity value remains unchanged
            if self.leverage > self.model.leverageRatio:
                self.portfolio = self.equity * self.model.leverageRatio
                self.model.e[self.unique_id] = self.portfolio
                self.deposit = self.portfolio - self.equity
                self.model.d[self.unique_id] = self.deposit
                self.leverage = self.model.leverageRatio
    
    def step(self):
        if not self.default:
            self.borrowRequest()
    

class bankingSystem(mesa.Model):
    def __init__(self, banksFile, 
                 leverageRatio, 
                 depositReserve,
                 num_borrowing,
                 sizeOfBorrowing,
                 num_banks, 
                 alpha = 0.99,
                 beta = 0.99,
                 concentrationParameter = None, 
                 fedRate = 0., 
                 portfolioReturnRate = 0., 
                 returnVolatiliy = 0.,
                 returnCorrelation = np.diag(np.ones(100)),
                 liquidityShockNum = 0,
                 shockSize = 0.0,
                 shockDuration=[-1,-1],
                 theta = np.zeros((2,1))):
        
        # interest rate
        self.fedRate = (fedRate+1)**(1/252) - 1
        # portfolio return rate
        self.portfolioReturnRate = (portfolioReturnRate+1)**(1/252) - 1
        # portfolio return volatility
        self.returnVolatiliy = returnVolatiliy/np.sqrt(252)
        # return correlation matrix
        self.returnCorrelation = returnCorrelation
        self.Cholesky = np.linalg.cholesky(returnCorrelation * self.returnVolatiliy**2)
        # number of liquidity shocks
        self.liquidityShockNum = liquidityShockNum 
        # size of the shock
        self.shockSize = shockSize
        # time of the shock
        self.shockDuration = shockDuration
        # asset recovery rate 
        self.alpha = alpha
        # interbank loan recovery rate
        self.beta = beta
        # learning parameter 
        self.theta = theta
        
        # read in banks equity capital
        banksData = pd.read_csv(banksFile).iloc[:num_banks,:]
        self.banks = banksData["bank"]
        self.N = num_banks
        self.leverageRatio = leverageRatio
        self.depositReserve = depositReserve
        self.num_borrowing = num_borrowing
        self.sizeOfBorrowing = sizeOfBorrowing
        # start with a uniform distribution of trust, using Dirichlet distribution as a conjugate prior
        # we also introduce a time decay factor for trust       
        if concentrationParameter is None:
            self.concentrationParameter = np.ones((self.N,self.N))
            np.fill_diagonal(self.concentrationParameter, 0.)
        else:
            self.concentrationParameter = concentrationParameter
        self.trustMatrix = self.concentrationParameter / (self.N-1.) / self.num_borrowing
        # liability matrix 
        self.L = np.zeros((self.N,self.N))
        # asset matrix
        self.e = (banksData["assets"].values).reshape(self.N,1)
        # deposit matrix
        self.d = banksData["assets"].values.reshape(self.N,1) * 0.8
        # create a schedule for banks
        self.schedule = mesa.time.RandomActivation(self)
        # target borrowing and lending amount based on risk aversion parameter
        self.targetBorrowingLending = np.zeros((self.N,1))
    
        # create banks and put them in schedule
        for i in range(self.N):
            a = Bank(i, self)
            a.deposit = self.d[i][0]
            a.portfolio = self.e[i][0]
            a.updateBlanceSheet()
            self.schedule.add(a)
            
        self.datacollector = mesa.DataCollector(
            model_reporters={"Trust Matrix": "trustMatrix",
                             "Liability Matrix": "L",
                             "Asset Matrix": "e"},
            agent_reporters={"PortfolioValue": "portfolio",
                                "Lending": "lending",
                                "Deposit": "deposit",
                                "Borrowing": "borrowing", 
                                "Equity": "equity",
                                "Default": "default",
                                "Leverage": "leverage"})
    
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def calculateBudget(self):
        # target investment ratio on the risky asset, (leverage, log(size))
        state = np.concatenate((self.e/(self.e - self.d + 0.01), np.log(1. + self.e)), axis=1)
        # standardized state
        state = (state - state.mean(axis = 0))/(state.std(axis = 0) + 0.01)
        random = np.random.randn(100,1)*0.1
        actions = state @ self.theta + random 
        gamma_lower = (self.portfolioReturnRate - self.fedRate) / self.returnVolatiliy**2 / 2
        gamma_upper = (self.portfolioReturnRate - self.fedRate) / self.returnVolatiliy**2 / 0.5
        gammas = self.sigmoid(actions) * (gamma_upper - gamma_lower) + gamma_lower
        targetRatio = (self.portfolioReturnRate - self.fedRate)/(gammas*(self.returnVolatiliy**2))
        # positive amount indicate borrowing and negative amount indicate lending
        self.targetBorrowingLending = ((targetRatio - 1) * (self.e-self.d * self.depositReserve))

    
    def updateTrustMatrix(self):
        # add time decay of concentration parameter
        self.concentrationParameter = self.concentrationParameter / self.concentrationParameter.sum(axis=1, keepdims=True) * (self.N - 1) * self.num_borrowing
        self.trustMatrix = self.concentrationParameter / (self.N - 1) / self.num_borrowing
            
    def clearingDebt(self): 
        # Returns the new portfolio value after clearing debt
        _, e = eisenbergNoe(self.L*(1+self.fedRate), self.e, self.alpha, self.beta)
        self.e = e
        insolventBanks = np.where(self.e - self.d <= 0)[0]
        # reset the Liabilities matrix after clearing debt
        self.L = np.zeros((self.N,self.N))
        if len(insolventBanks) > 0:
            self.concentrationParameter[:,insolventBanks] = 0
            self.e[insolventBanks] = 0
        for agent in self.schedule.agents:
            agent.reset()

    def returnOnPortfolio(self):
        # Return on the portfolio:
        self.e += (self.e - self.d*self.depositReserve) * (self.portfolioReturnRate + (self.Cholesky @ np.random.randn(self.N,1)))
           
    def liquidityShock(self):
        # liquidity shock to banks portfolio
        if self.schedule.time >= self.shockDuration[0] and self.schedule.time <= self.shockDuration[1]:
            if self.liquidityShockNum > 0:
                exposedBank = np.random.choice(self.N, self.liquidityShockNum,replace=False)
                # set the bank's equity to drop
                self.e[exposedBank] -= (self.e[exposedBank] - 
                                        self.d[exposedBank] * self.depositReserve)*self.shockSize
    
    def correlatedShock(self):
        # liquidity shock to banks portfolio
        if self.schedule.time >= self.shockDuration[0] and self.schedule.time <= self.shockDuration[1]:
            # set the bank's equity to drop
            self.e -= (self.e - self.d * self.depositReserve)*(self.shockSize + 5*self.Cholesky @ np.random.randn(self.N,1))

    def simulate(self):
        # only one banks remains solvent
        self.updateTrustMatrix()
        self.calculateBudget()
        self.schedule.step()
        self.returnOnPortfolio()
        #self.liquidityShock()
        self.correlatedShock()
        self.datacollector.collect(self)
        self.clearingDebt()
