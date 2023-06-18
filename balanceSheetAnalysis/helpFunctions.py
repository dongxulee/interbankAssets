import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Read cached data
years = np.arange(2001,2023)
dirName = "pickleData/"
# show all the file names under the folder
files = sorted(os.listdir(dirName))
data = [pd.read_pickle(dirName + f) for f in files if f.startswith("20") and f.endswith(".pkl")]
varNamesOverYears = pickle.load(open(dirName + "varNamesOverYears.pkl", "rb"))
varNamesOverYearsPlus = pickle.load(open(dirName + "varNamesOverYearsPlus.pkl", "rb"))

def loadData():
    # Read cached data
    years = np.arange(2001,2024)
    dirName = "pickleData/"
    # show all the file names under the folder
    files = sorted(os.listdir(dirName))
    data = [pd.read_pickle(dirName + f) for f in files if f.startswith("20") and f.endswith(".pkl")]
    varNamesOverYears = pickle.load(open(dirName + "varNamesOverYears.pkl", "rb"))
    varNamesOverYearsPlus = pickle.load(open(dirName + "varNamesOverYearsPlus.pkl", "rb")) 
    return years, data, varNamesOverYears, varNamesOverYearsPlus

def interestingVar(keyWords, variables):
    '''
        print the variables that contain all the key words
    '''
    interestingVars = []
    for var in variables:
        # if var contains loan or debts or total or asset or liability, then print it out
        if all([word.lower() in var.lower() for word in keyWords]):
            interestingVars.append(var)
    return interestingVars

def interestingVar(keyWords, variables):
    '''
        print the variables that contain all the key words
    '''
    interestingVars = []
    for var in variables:
        # if var contains loan or debts or total or asset or liability, then print it out
        if all([word.lower() in var.lower() for word in keyWords]):
            interestingVars.append(var)
    return interestingVars

def searchVar(keyWords, printOut=2, show = True, varNamesOverYears = varNamesOverYears, varNamesOverYearsPlus = varNamesOverYearsPlus):
    '''
        print the variables that contain all the key words, use this function along with Fed data dictionary
        https://www.federalreserve.gov/apps/mdrm/data-dictionary
    '''
    # search for interesting variables
    result = []
    for i,varName in enumerate(zip(varNamesOverYears, varNamesOverYearsPlus)):
        var1 = interestingVar(keyWords, varName[0])
        var2 = interestingVar(keyWords, varName[1])
        if show:
            print(f"year {i+2001}:")
        if printOut==1:
            if show:
                [print(v) for v in var1]
            result.append(var1)
        else:
            if show:
                [print(v) for v in var2]
            result.append(var2)
    # find the common variables over the years       
    commonVarsOverTheYears = set(result[0])
    for re in result[1:]:
        commonVarsOverTheYears = commonVarsOverTheYears.intersection(set(re))
    return list(commonVarsOverTheYears)
            
def call(vars, year, data = data, varNamesOverYears=varNamesOverYearsPlus):
    '''
        vars: list of variables, a variable seems to have multiple names 
        year: year of data
        return: a dataframe of a single aggregate variable
    '''
    dd = data[year - 2001]
    indeces = [i for i, v in enumerate(varNamesOverYears[year - 2001]) if v in vars]
    df = pd.DataFrame()
    df["Financial Institution Name"] = dd["Financial Institution Name"]
    df["Reporting Period End Date"] = dd["Reporting Period End Date"]
    df[vars[0]] = dd.iloc[:, indeces].fillna(0).sum(axis=1) 
    return vars[0], df

def aggregateSumOverYears(des, name, beginYear, endYear):
    collectVars = []
    for year in range(beginYear,endYear+1):
        if type(des) == str:
            var, ddd = call([des], year)
        else:
            var, ddd = call(des, year)
        collectVars.append(ddd.groupby("Financial Institution Name")[var].mean().sum())
    collection = pd.DataFrame(np.array([range(beginYear, endYear+1), collectVars]).T, columns=["DATE", name])
    collection["DATE"] = pd.to_datetime(collection["DATE"], format="%Y")
    collection["DATE"] = collection["DATE"].dt.year
    collection.set_index("DATE", inplace=True)
    return collection

def comparisonPlotsOverYears(df1, df2, label1, label2, ylabel1, ylabel2):
    fig, ax = plt.subplots()
    ax.plot(df1.index, df1[df1.columns[0]],"r", label=label1)
    ax.set_ylabel(ylabel1, color='r', fontsize=12)
    ax.legend(bbox_to_anchor=(1, 1.1))
    ax1 = ax.twinx()
    ax.set_xlabel('Year', fontsize=10)
    ax1.plot(df2.index, df2[df2.columns], 'b', label=label2)
    ax1.set_ylabel(ylabel2, color='b', fontsize=12)
    ax1.legend(bbox_to_anchor=(0.15, 1.1))
    plt.xticks(range(2001,2023))
    plt.show()