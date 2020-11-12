# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data = pd.read_csv(path)
data['Gender'].replace('-','Agender',inplace=True)
gender_count = data['Gender'].value_counts()
print(gender_count)
height = [413,157,24]
plt.bar(gender_count,height,color='red',width = 1)
#Code starts here 




# --------------
#Code starts here
alignment = data['Alignment'].value_counts()
print(alignment)
print(alignment.index)
print(alignment.values)
plt.pie(alignment.values,labels=alignment.index)


# --------------
#Code starts here
# For Strength and Combat Columns
sc_df = data[['Strength','Combat']].copy()
sc_df
sc_covariance = sc_df.cov().iloc[0,1]
print(sc_covariance)
sc_strength = sc_df.Strength.std()
print(sc_strength)
sc_combat = sc_df.Combat.std()
print(sc_combat)
sc_pearson = sc_covariance/(sc_strength*sc_combat)
print(sc_pearson)

# For Intelligence and Combat Columns
ic_df = data[['Intelligence','Combat']].copy()
ic_df
ic_covariance = ic_df.cov().iloc[0,1]
print(ic_covariance)
ic_intelligence = ic_df.Intelligence.std()
print(ic_intelligence.round(2))
ic_combat = ic_df.Combat.std()
print(ic_combat)
ic_pearson = ic_covariance/(ic_intelligence*ic_combat)
print(ic_pearson)





# --------------
#Code starts here
total_high = data['Total'].quantile(q=0.99)
print(total_high)
super_best = data[data['Total'] > total_high]
super_best_names = [x for x in super_best['Name']]
print(super_best_names) 



# --------------
#Code starts here
fig,(ax_1,ax_2,ax_3) = plt.subplots(3)
ax_1.boxplot(super_best['Intelligence'])
ax_2.boxplot(super_best['Speed'])
ax_3.boxplot(super_best['Power'])


