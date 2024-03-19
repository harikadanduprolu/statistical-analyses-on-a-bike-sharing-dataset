import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
df = pd.read_csv(r"C:\Users\niece\Downloads\yulu_bike_sharing_dataset.csv")
data_table = pd.crosstab(df['season'], df['weather'])
print("Observed values:")
data_table

val = stats.chi2_contingency(data_table)
print(val)
print()
print("***********************************************")
Expected_values = val[3]
print(f'Expected_values : {val[3]}')
print()
print("***********************************************")


nrows, ncols = 4, 4
dof = (nrows-1)*(ncols-1)
print(f"Degrees of freedom: {dof}")
print()
print("***********************************************")
alpha = 0.05
chi_sqr = sum([(o-e)**2/e for o, e in zip(data_table.values,Expected_values)])
chi_sqr_statistic = chi_sqr[0] + chi_sqr[1]
print(f"Chi-square test statistic: {chi_sqr_statistic}")
print()
print("***********************************************")


critical_val = stats.chi2.ppf(q=1-alpha, df=dof)
print(f"Critical value: {critical_val}")
print()
print("***********************************************")

p_val = 1-stats.chi2.cdf(x=chi_sqr_statistic, df=dof)
print(f"P-value: {p_val}")
print()
print("***********************************************")

if p_val <= alpha:
  print("Since p-value is less than the alpha 0.05 we reject Null Hypothesis. This indicates weather is dependent on the season.")
else:
  print("Since p-value is greater than the alpha 0.05 we do not reject the Null Hypothesis")

data_group1 = df[df['workingday']==0]['count'].values
data_group2 = df[df['workingday']==1]['count'].values
print(np.var(data_group1), np.var(data_group2))
np.var(data_group2)// np.var(data_group1)

stats.ttest_ind(a=data_group1, b=data_group2, equal_var=True)

df["weather"].unique()

df["weather"].value_counts()

sns.boxplot(x='weather', y='count', data=df)
plt.show()

count_g1 = df[df["weather"]==1]["count"]
count_g2 = df[df["weather"]==2]["count"]
count_g3 = df[df["weather"]==3]["count"]
count_g4 = df[df["weather"]==4]["count"]

a,b,c,d = [round(count_g1.mean(), 2),round(count_g2.mean(),2),round(count_g3.mean(),2),round(count_g4.mean(),2)]

print(a, end= " ")
print(b, end= " ")
print(c, end= " ")
print(d)

from scipy.stats import f_oneway, kruskal   

# H0: All weather's have same number of cycles rented.
# Ha: Atleast one or more weather conditions have different number of cycles rented.

f_stats, p_value = f_oneway(count_g1,count_g2,count_g3,count_g4)
print(f"p_value : {p_value}")
print()

if p_value < 0.05:
    print("Reject H0")
    print("Different weathers have different number of cycles rented")
else:
    print("Fail to reject H0 or accept H0")
    print("All weather's have same number of cycles rented.")

df["season"].unique()

df["season"].value_counts()

sns.boxplot(x='season', y='count', data=df)
plt.show()

coun_g1 = df[df["season"]==1]["count"]
coun_g2 = df[df["season"]==2]["count"]
coun_g3 = df[df["season"]==3]["count"]
coun_g4 = df[df["season"]==4]["count"]

a,b,c,d = [round(coun_g1.mean(), 2),round(coun_g2.mean(),2),round(coun_g3.mean(),2),round(coun_g4.mean(),2)]

print(a, end= " ")
print(b, end= " ")
print(c, end= " ")
print(d)

from scipy.stats import f_oneway, kruskal   # Numeric Vs categorical for many categories

# H0: All seasons's have same number of cycles rented.
# Ha: Atleast one or more seasons  have different number of cycles rented.

f_stats, p_value = f_oneway(coun_g1,coun_g2,coun_g3,coun_g4)
print(f"p_value : {p_value}")
print()

if p_value < 0.05:
    print("Reject H0")
    print("Different seasons have different number of cycles rented")
else:
    print("Fail to reject H0 or accept H0")
    print("All seasons have same number of cycles rented.")

import numpy as np 
import statsmodels.api as sm 
import pylab as py 

a = [count_g1,count_g2,count_g3,count_g4]

fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

sm.qqplot(a[0], line = "s", ax = axis[0,0])
sm.qqplot(a[1], line = "s", ax = axis[0,1])
sm.qqplot(a[2], line = "s", ax = axis[1,0])
sm.qqplot(a[3], line = "s", ax = axis[1,1])

plt.show()

a = [count_g1,count_g2,count_g3,count_g4]

fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

index = 0
for row in range(2):
  for col in range(2):
    sns.histplot(a[index], ax=axis[row, col], kde=True)
    index += 1
plt.show()

b = [coun_g1,coun_g2,coun_g3,coun_g4]

fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

sm.qqplot(b[0], line = "s", ax = axis[0,0])
sm.qqplot(b[1], line = "s", ax = axis[0,1])
sm.qqplot(b[2], line = "s", ax = axis[1,0])
sm.qqplot(b[3], line = "s", ax = axis[1,1])

plt.show()

b = [coun_g1,coun_g2,coun_g3,coun_g4]

fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

index = 0
for row in range(2):
  for col in range(2):
    sns.histplot(b[index], ax=axis[row, col], kde=True)
    index += 1
plt.show()
count_g1_subset = count_g1.sample(100)

# H0: Data is Gaussian
# Ha: Data is not Gaussian
from scipy.stats import shapiro
from scipy.stats import levene

test_stat, p_value = shapiro(count_g1_subset)
print(p_value)
if p_value<0.05:
    print("Data is not gaussian")
else:
    print("Data is gaussian")

coun_g1_subset = coun_g1.sample(100)

# H0: Data is Gaussian
# Ha: Data is not Gaussian
from scipy.stats import shapiro
from scipy.stats import levene

test_stat, p_value = shapiro(coun_g1_subset)
print(p_value)
if p_value<0.05:
    print("Data is not gaussian")
else:
    print("Data is gaussian")

sns.histplot(data= df, x="count", hue= "weather", color = "o")
plt.show()
#Equal variance: Levene's Test
#Null Hypothesis: Variances is similar in different weather and season.
#Alternate Hypothesis: Variances is not similar in different weather and season
# H0: Variances are equal
# Ha: Variances are not equal
#Significance level (alpha): 0.05
levene_stat, p_value = levene(count_g1,count_g2,count_g3,count_g4)
print(f'p-value : {p_value}')
if p_value < 0.05:
    print("Reject the null hypthesis.Variances are not similar.")
else:
    print("Variance are similar.")

sns.histplot(data= df, x="count", hue= "season", color = "o")
plt.show()

# H0: Variances are equal
# Ha: Variances are not equal
levene_stat, p_value = levene(coun_g1,coun_g2,coun_g3,coun_g4)
print(f'p-value : {p_value}')
if p_value < 0.05:
    print("Reject the null hypthesis. Variances are not similar.")
else:
    print("Variance are similar.")


kruskal_stat, p_value = stats.kruskal(count_g1,count_g2,count_g3,count_g4)
print(f"p_value : {p_value}")

if p_value<0.05:
  print("Since p-value is less than 0.05, we reject the null hypothesis")
  print('Different weather have different number of cycles rented.')
else :
  print("Failes to reject null hypothesis. All weathers has same number of cycles rented.")

kruskal_stat, p_value = stats.kruskal(coun_g1,coun_g2,coun_g3,coun_g4)
print(f"p_value : {p_value}")

if p_value<0.05:
  print("Since p-value is less than 0.05, we reject the null hypothesis")
  print('Different weather have different number of cycles rented.')
else :
  print("Failed to reject null hypothesis. All weathers has same number of cycles rented.")