import glob
import pandas as pd
file = '../Task4_inputs.csv'
data_for_images=pd.read_csv(file,encoding = "ISO-8859-1",header=None)
str=data_for_images.iloc[0,0]
str1=data_for_images.iloc[0,1]
inputFile ='../img/'+str+'*'+str1+'*.csv'
for f in glob.glob(inputFile):
    print(f)