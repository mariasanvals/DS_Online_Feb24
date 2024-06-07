import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_tips = sns.load_dataset('tips')
print(df_tips.head(40))

fig = plt.figure()
sns.histplot(data=df_tips,x='total_bill',kde=True)
fig.savefig('total_bill_hist.png')

fig = plt.figure()
sns.countplot(data=df_tips,x='time',hue='sex')
fig.savefig('countplot_time_sex.png')

fig = plt.figure()
sns.histplot(data=df_tips,x='total_bill',hue='sex',bins=20,kde=True)
fig.savefig('total_bill_by_sex.png')