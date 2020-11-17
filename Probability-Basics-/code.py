# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv(path)
#print(df.info())
p_a = ((df['fico'] > 700).sum())/len(df)
print(p_a)
p_b = ((df['purpose'] == 'debt_consolidation').sum())/len(df)
print(p_b)
df1 = df[df['purpose']== 'debt_consolidation']
p_a_b = df1[df1['fico'].astype(float) >700].shape[0]/df1.shape[0]
print(p_a_b)
result = p_a_b == p_a
print(result)
# code ends here


# --------------
# code starts here
prob_lp = (df['paid.back.loan'] == 'Yes').sum()/len(df)
print(prob_lp)
prob_cs = (df['credit.policy'] == 'Yes').sum()/len(df)
print(prob_cs)
new_df = df[df['paid.back.loan'] == 'Yes']
prob_pd_cs = (new_df['credit.policy'] == 'Yes').sum()/len(new_df)
print(prob_pd_cs)
bayes = (prob_pd_cs*prob_lp)/prob_cs
print(bayes)
# code ends here


# --------------
# code starts here
plt.bar(df['purpose'],df['purpose'].index)
df1 = df[df['paid.back.loan'] == 'No']
df1
plt.bar(df1['purpose'],df1['purpose'].index)
# code ends here


# --------------
# code starts here
inst_median = df['installment'].median()
print(inst_median)
inst_mean = df['installment'].mean()
print(inst_mean)
plt.hist(df['installment'])
plt.hist(df['log.annual.inc'])
# code ends here


