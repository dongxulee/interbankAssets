import os
import numpy as np
import pandas as pd
import pickle

# read in all the files under the folder
dirName = "balanceSheets/"
# show all the file names under the folder
files = sorted(os.listdir(dirName))
data = []
# merge same years data
previousYear = 0
varNamesOverYears = []
for fileName in files:
    print(fileName)
    if fileName.endswith(".csv"):
        break
    thisYear = int(fileName.split()[-3][:4])
    ## read in first two lines of the file 
    with open(dirName + fileName) as f:
        line1 = f.readline()
        # line2 are code names here we are using out source file to map code to name
        line2 = f.readline()
        headers1 = line1.split("\t")
        headers2 = line2.split("\t")
        assert len(headers1) == len(headers2)
        names = [h2 if h2 != '' else h1 for h1, h2 in zip(headers1, headers2)]
        # use headers1 for now, sine headers2 has duplicate names
        df = pd.read_csv('balanceSheets/' + fileName,  sep="\t", skiprows=2, header=None, engine="python")
        df.columns = headers1
            # merge same years data
        if thisYear == previousYear:
            identifier = np.intersect1d(df.columns, data[-1].columns)
            data[-1] = pd.merge(data[-1], df, on=list(identifier), how='outer')
            extraNames = [n for n in names if n not in identifier]
            varNamesOverYears[-1] = varNamesOverYears[-1] + extraNames
            # make sure the name matches the identifier
            assert(len(data[-1].columns) == len(varNamesOverYears[-1]))
        else:
            data.append(df)        
            varNamesOverYears.append(names)
        previousYear = thisYear
        
# how many years of data do we have 
print(f"{len(data)} years of data")

# how many years of data do we have 
print(f"{len(data)} years of data")

print("(# samples, # variables)")
for dd in data:
    print(dd.shape)
    
for dd, year in zip(data, range(2001, 2024)):
    dd.to_pickle("pickleData/" + str(year) + ".pkl")

# save the variable names over years
pickle.dump(varNamesOverYears, open("pickleData/varNamesOverYears.pkl", "wb"))

# map code to name
codeMap = pd.read_csv('balanceSheets/codeMap.csv', skiprows=1)
codeMap["code"] = codeMap["Mnemonic"].astype(str) + codeMap["Item Code"].astype(str)
# mapping from code to var 
code_to_var = dict(zip(codeMap['code'], codeMap['Item Name']))
# some codes are missing in the codeMap
code_to_var["RCFD0071"] = "INTEREST-BEARING BALANCES"
code_to_var["RCFD0081"] = "NONINTEREST-BEARING BALANCES AND CURRENCY AND COIN"
code_to_var["RCFD0426"] = "OTHER IDENTIFIABLE INTANGIBLE ASSETS"
code_to_var["RCON0071"] = "INTEREST-BEARING BALANCES"
code_to_var["RCON0081"] = "NONINTEREST-BEARING BALANCES AND CURRENCY AND COIN"
code_to_var["RCON0426"] = "OTHER IDENTIFIABLE INTANGIBLE ASSETS"
code_to_var["RIAD0093"] = "SAVINGS DEPOSITS (INCLUDING MMDAS)"


varNamesOverYearsPlus = []
for dd in data:
    varNamesOverYearsPlus.append([code_to_var[code] if code in code_to_var else code for code in dd.columns])

pickle.dump(varNamesOverYearsPlus, open("pickleData/varNamesOverYearsPlus.pkl", "wb"))


''' In case we want to get quarterly data


    date = []
    for d in data:
        date = np.append(date,d['Reporting Period End Date'].unique())
    date = sort(date)
    date
    
    
'''