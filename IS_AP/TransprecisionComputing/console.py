import pickle as pkl
import pandas as pd

filein = r'C:\RepositoryGIT\AlmaMaterStudiorum\IntelligentSystems_ProjectWork\IS_AP\data\input\tpc\exp_results_correlation_10000.pickle'
fileout = r'C:\RepositoryGIT\AlmaMaterStudiorum\IntelligentSystems_ProjectWork\IS_AP\data\input\tpc\exp_results_correlation_10000.csv'

if False :

    with open(filein, "rb") as f:
        object = pkl.load(f)
    
    df = pd.DataFrame(object)
    df.to_csv(fileout)



df = pd.read_csv(fileout)

# iterating the columns
for col in df.columns:
    row0 = df[col][0]
    row1 = df[col][1]
    message = '{col} : {row0}  {row1}'.format(col=col,row0=row0,row1=row1)
    print(message)


print(df.iloc[0])

print(df.iloc[1])




