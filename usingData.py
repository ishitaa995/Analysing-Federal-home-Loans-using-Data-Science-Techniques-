import pandas as pd
import matplotlib.pyplot as plt
df = pd. read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/LoanData.csv")
df.drop(['LoanNumber', 'AssignedID','Coop','Program','AcquDate','MortDate','Bed1','Bed2','Bed3','Bed4','Aff1','Aff2','Aff3','Aff4','Occup','NumUnits','RentUt1','RentUt2','RentUt3','Product','RentUt4','Rent1','Rent2','Rent3','Rent4','FeatureID','SellType','Seller','SellCity','SellSt','CICA','LienStatus','FedFinStbltyPlan','GSEREO'], axis=1, inplace=True)
df['MSA'].isnull().any()

# sample_df = df.groupby('Year').apply(lambda x: x.sample(frac=0.01))
# sample_df .plot()
# sample_df.to_csv("    C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/LoanData_Smapling.csv",index=False)