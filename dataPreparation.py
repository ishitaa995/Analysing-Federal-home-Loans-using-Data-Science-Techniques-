import pandas as pd

# twenty_12 = pd. read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/y_2021.csv")
# target=["LoanNumber","Program"]
# count=0
# def isNaN(num):
#     return num!= num
# length=len(twenty_12)
#
# for element in twenty_12[target]:
#     if type(element) == float and pd.isna(element):
#         twenty_12[element]=0
#         count = count + 1
#
# twenty_12[target]=twenty_12[target].fillna(0)
# twenty_12[target]=twenty_12[target].astype('int')
#
# twenty_12.to_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/y_2021.csv",index=False)

df_09 = pd. read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/y_2009.csv")
df_10= pd. read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/y_2010.csv")
df_11= pd. read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/y_2011.csv")
df_12= pd. read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/y_2012.csv")
df_13= pd. read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/y_2013.csv")
df_14= pd. read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/y_2014.csv")
df_15= pd. read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/y_2015.csv")
df_16= pd. read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/y_2016.csv")
df_17= pd. read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/y_2017.csv")
df_18= pd. read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/y_2018.csv")
df_19= pd. read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/y_2019.csv")
df_20= pd. read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/y_2020.csv")
df_21= pd. read_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/y_2021.csv")
frames = [df_09, df_10,df_11,df_12,df_13,df_14,df_15,df_16,df_17,df_18,df_19,df_20,df_21]
result = pd.concat(frames)
# dummy=[df_09, df_10,df_11,df_12,df_13,df_14,df_15,df_16,df_17,df_18,df_19,df_20,df_21]
# result = pd.concat(dummy)
result.to_csv("C:/Users/kavya/OneDrive/Desktop/Spring2023_890/LoanFHLBData/original/LoanData.csv",index=False)