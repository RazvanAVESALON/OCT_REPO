from multiprocessing.sharedctypes import Value
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
df = pd.read_csv(r"D:\ai intro\OCT\OCT_file\Stats.csv")
data= pd.read_csv(r'D:\ai intro\OCT\OCT_file\Statistica.csv')
     
     
Dice=pd.read_csv(r'D:\ai intro\OCT\OCT_REPO\DICE_ADDED_PREDICTII_v2.csv')      
#plt.hist(data['aria'],[0.10000,20000,30000,40000,50000,60000,70000,80000,90000])  
# plt.hist(data['aria'], 20)  
# plt.title('ARIA')
# plt.xlabel("Value")
# plt.ylabel("Count")
# plt.savefig(r"D:\ai intro\OCT\OCT_REPO\Histograma-Arie")   




print (Dice['dice_calculat_de_mine'])
plt.hist(Dice['dice_calculat_de_mine'] ,[0,0.2,0.4,0.6,0.8,1])

plt.title("DICE INDEX ")
plt.xlabel("Value")
plt.ylabel("Count")
plt.savefig(r'D:\ai intro\OCT\OCT_REPO\Histograma_Dice'  )
           




# plt.hist(df['n_total_slices'],[0,5,10,15,20,25,30])
# plt.savefig(r"D:\ai intro\OCT\OCT_file\Histograma")