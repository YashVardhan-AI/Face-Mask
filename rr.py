import pandas as pd
import numpy as np
my_np1=np.array([[30,40,50,45],
                 [50,60,50,55]])
my_names=np.array(['Alex','Ron','Jack','King'])
my_pd=pd.DataFrame(data=[my_names,my_np1[0],my_np1[1]]).T
my_pd.columns=['NAMES','MATH','ENGLISH']
my_pd['Total']=my_pd['MATH'] + my_pd['ENGLISH']
print(my_pd)
print(my_pd['Total'][0])