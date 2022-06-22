from multiprocessing.sharedctypes import Value
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
df = pd.read_csv(r"D:\ai intro\OCT\OCT_file\Stats.csv")
data= pd.read_csv(r'D:\ai intro\OCT\OCT_file\Statistica.csv')
     
     
Dice=pd.read_csv(r'D:\ai intro\OCT\OCT_REPO\DICE_ADDED.csv')      
#plt.hist(data['aria'],[0.10000,20000,30000,40000,50000,60000,70000,80000,90000])  
# plt.hist(data['aria'], 20)  
# plt.title('ARIA')
# plt.xlabel("Value")
# plt.ylabel("Count")
# plt.savefig(r"D:\ai intro\OCT\OCT_REPO\Histograma-ARIE")   




print (Dice['dice'])
plt.hist(Dice['dice'] ,[-1,0,1])

plt.title("DICE INDEX ")
plt.xlabel("Value")
plt.ylabel("Count")
plt.savefig(r'D:\ai intro\OCT\OCT_REPO\HISTOGRAMA_DICE'  )
           




# plt.hist(df['n_total_slices'],[0,5,10,15,20,25,30])
# plt.savefig(r"D:\ai intro\OCT\OCT_file\Histograma")