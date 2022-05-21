#!/usr/bin/env python
# coding: utf-8

# In[143]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter as c # return counts
import seaborn as sns #used for data Visualization
import matplotlib.pyplot as plt
#import missingno as msno #finding missing values
from sklearn.model_selection import train_test_split #splits data in random train and test array
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error#model performance
import pickle #Python object hierarchy is converted into a byte stream,
from sklearn.linear_model import LinearRegression #Regresssion ML algorithm


# In[144]:


data = pd.read_csv('Indicators.csv')
# #data.shape
#
#
# # In[145]:
#
#
# #data.head(10)
#
#
# # In[146]:
#
#
# #data.Value.round(2)
#
#
# # In[147]:
#
#
# countries = data['CountryName'].unique().tolist()
# len(countries)
#
#
# # In[148]:
#
#
# # How many unique country codes are there ? (should be the same #)
# countryCodes = data['CountryCode'].unique().tolist()
# len(countryCodes)
#
#
# # In[149]:
#
#
# # How many unique indicators are there ? (should be the same #)
# indicators = data['IndicatorName'].unique().tolist()
# len(indicators)
#
#
# # In[150]:
#
#
# # How many years of data do we have ?
# years = data['Year'].unique().tolist()
# len(years)
#
#
# # In[151]:
#
#
# print(min(years)," to ",max(years))
#
#
# # In[153]:
#
#
# # select CO2 emissions for the United States
# hist_indicator = 'CO2 emissions \(metric'
# hist_country = 'USA'
#
# mask1 = data['IndicatorName'].str.contains(hist_indicator)
# mask2 = data['CountryCode'].str.contains(hist_country)
#
# # stage is just those indicators matching the USA for country code and CO2 emissions over time.
# stage = data[mask1 & mask2]
#
#
# # In[154]:
#
#
# stage.head()
#
#
# # In[155]:
#
#
# # get the years
# years = stage['Year'].values
# # get the values
#
# co2 = stage['Value'].values
#
# # create
# plt.bar(years,co2)
# plt.show()
#
#
# # In[156]:
#
#
# # switch to a line plot
# plt.plot(stage['Year'].values, stage['Value'].values)
#
# # Label the axes
# plt.xlabel('Year')
# plt.ylabel(stage['IndicatorName'].iloc[0])
#
# #label the figure
# plt.title('CO2 Emissions in USA')
#
# # to make more honest, start they y axis at 0
# plt.axis([1959, 2011,0,25])
# #plt.plot(stage['Year'].values, stage['Value'].values)
#
# plt.show()
#
#
# # In[157]:
#
#
# # If we want to just include those within one standard deviation fo the mean, we could do the following
# # lower = stage['Value'].mean() - stage['Value'].std()
# # upper = stage['Value'].mean() + stage['Value'].std()
# # hist_data = [x for x in stage[:10000]['Value'] if x>lower and x<upper ]
#
# # Otherwise, let's look at all the data
# hist_data = stage['Value'].values
#
#
# # In[158]:
#
#
# print(len(hist_data))
#
#
# # In[159]:
#
#
# # the histogram of the data
# plt.hist(hist_data, 10, normed=False, facecolor='green')
#
# plt.xlabel(stage['IndicatorName'].iloc[0])
# plt.ylabel('# of Years')
# plt.title('Histogram Example')
#
# plt.grid(True)
#
# plt.show()
#
#
# # In[160]:
#
#
# # select CO2 emissions for all countries in 2011
# hist_indicator = 'CO2 emissions \(metric'
# hist_year = 2011
#
# mask1 = data['IndicatorName'].str.contains(hist_indicator)
# mask2 = data['Year'].isin([hist_year])
#
# # apply our mask
# co2_2011 = data[mask1 & mask2]
# co2_2011.head()
#
#
# # For how many countries do we have CO2 per capita emissions data in 2011
#
# # In[161]:
#
#
# print(len(co2_2011))
#
#
# # In[162]:
#
#
# # let's plot a histogram of the emmissions per capita by country
#
# # subplots returns a touple with the figure, axis attributes.
# fig, ax = plt.subplots()
#
# ax.annotate("USA",
#             xy=(18, 5), xycoords='data',
#             xytext=(18, 30), textcoords='data',
#             arrowprops=dict(arrowstyle="->",
#                             connectionstyle="arc3"),
#             )
#
# plt.hist(co2_2011['Value'], 10, normed=False, facecolor='green')
#
# plt.xlabel(stage['IndicatorName'].iloc[0])
# plt.ylabel('# of Countries')
# plt.title('Histogram of CO2 Emissions Per Capita')
#
# #plt.axis([10, 22, 0, 14])
# plt.grid(True)
#
# plt.show()
#
#
# # So the USA, at ~18 CO2 emissions (metric tons per capital) is quite high among all countries.
# #
# # An interesting next step, which we'll save for you, would be to explore how this relates to other industrialized nations and to look at the outliers with those values in the 40s!
#
# # ### Relationship between GDP and CO2 Emissions in USA
#
# # In[163]:
#
#
# # select GDP Per capita emissions for the United States
# hist_indicator = 'GDP per capita \(constant 2005'
# hist_country = 'USA'
#
# mask1 = data['IndicatorName'].str.contains(hist_indicator)
# mask2 = data['CountryCode'].str.contains(hist_country)
#
# # stage is just those indicators matching the USA for country code and CO2 emissions over time.
# gdp_stage = data[mask1 & mask2]
#
# #plot gdp_stage vs stage
#
#
# # In[164]:
#
#
# gdp_stage.head()
#
#
# # In[165]:
#
#
# stage.head(2)
#
#
# # In[166]:
#
#
# # switch to a line plot
# plt.plot(gdp_stage['Year'].values, gdp_stage['Value'].values)
#
# # Label the axes
# plt.xlabel('Year')
# plt.ylabel(gdp_stage['IndicatorName'].iloc[0])
#
# #label the figure
# plt.title('GDP Per Capita USA')
#
# # to make more honest, start they y axis at 0
# #plt.axis([1959, 2011,0,25])
#
# plt.show()
#
#
# # So although we've seen a decline in the CO2 emissions per capita, it does not seem to translate to a decline in GDP per capita
#
# # ### ScatterPlot for comparing GDP against CO2 emissions (per capita)
# #
# # First, we'll need to make sure we're looking at the same time frames
#
# # In[167]:
#
#
# print("GDP Min Year = ", gdp_stage['Year'].min(), "max: ", gdp_stage['Year'].max())
# print("CO2 Min Year = ", stage['Year'].min(), "max: ", stage['Year'].max())
#
#
# # We have 3 extra years of GDP data, so let's trim those off so the scatterplot has equal length arrays to compare (this is actually required by scatterplot)
#
# # In[168]:
#
#
# gdp_stage_trunc = gdp_stage[gdp_stage['Year'] < 2012]
# print(len(gdp_stage_trunc))
# print(len(stage))
#
#
# # In[169]:
#
#
# get_ipython().run_line_magic('matplotlib', 'inline')
# import matplotlib.pyplot as plt
#
# fig, axis = plt.subplots()
# # Grid lines, Xticks, Xlabel, Ylabel
#
# axis.yaxis.grid(True)
# axis.set_title('CO2 Emissions vs. GDP \(per capita\)',fontsize=10)
# axis.set_xlabel(gdp_stage_trunc['IndicatorName'].iloc[10],fontsize=10)
# axis.set_ylabel(stage['IndicatorName'].iloc[0],fontsize=10)
#
# X = gdp_stage_trunc['Value']
# Y = stage['Value']
#
# axis.scatter(X, Y)
# plt.show()
#
#
# # This doesn't look like a strong relationship.  We can test this by looking at correlation.
#
# # In[170]:
#
#
# np.corrcoef(gdp_stage_trunc['Value'],stage['Value'])
#
#
# # In[171]:
#
#
# data.info() #info will give you a summary of dataset
#
#
# # In[172]:
#
#
# data.describe()  # returns important values for continous column data
#
#
# # In[173]:
#
#
# np.unique(data.dtypes,return_counts=True)
#
#
# # In[174]:


cat=data.dtypes[data.dtypes=='O'].index.values
cat


# In[175]:


for i in cat:
    print("Column :",i)
    print('count of classes : ',data[i].nunique())
    print(c(data[i]))
    print('*'*120)


# In[176]:


data.dtypes[data.dtypes!='O'].index.values


# In[177]:


data.isnull().any()#it will return true if any columns is having null values


# In[178]:


data.isnull().sum() #used for finding the null values


# In[179]:


data=data[data['CountryCode'].str.contains("USA|SGP|IND|ARB|BRB")]


# In[207]:


data.head()


# In[180]:


data1=data.copy()
from sklearn.preprocessing import LabelEncoder #imorting the LabelEncoding from sklearn
x='*'
for i in cat:#looping through all the categorical columns
    print("LABEL ENCODING OF:",i)
    LE = LabelEncoder()#creating an object of LabelEncoder
    print(c(data[i])) #getting the classes values before transformation
    data[i] = LE.fit_transform(data[i]) # trannsforming our text classes to numerical values
    print(c(data[i])) #getting the classes values after transformation
    print(x*100)


# In[181]:


data.head()


# In[182]:


corr = data.corr() #perform correlation between all continous features
plt.subplots(figsize=(16,16));
sns.heatmap(corr, annot=True, square=True) #plotting heatmap of correlations
plt.title("Correlation matrix of numerical features")
plt.tight_layout()
plt.show()


# In[183]:


plt.figure(figsize=(16,5))
corr["Value"].sort_values(ascending=True)[:-1].plot(kind="barh")


# In[184]:


data.head()


# In[210]:


data.info()


# In[214]:


x = data.drop(['Value','IndicatorCode'],axis=1) #independet features
x=pd.DataFrame(x)
y = data['Value'] #dependent feature
y=pd.DataFrame(y)


# In[215]:


x.head()


# In[216]:


y.head()


# In[217]:


data.head()


# In[218]:


type(x)


# In[191]:


type(y)


# In[192]:


data['CountryCode'].unique()


# In[ ]:





# In[219]:


data.shape


# In[220]:


data["CountryCode"].unique()


# In[221]:


data['CountryName'].unique()


# In[222]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
print(x_train.shape)
print(x_test.shape)


# In[223]:


from sklearn.ensemble import RandomForestRegressor
rand=RandomForestRegressor(n_estimators=10,random_state=52)
rand.fit(x_train,y_train)


# In[224]:


x_test


# In[225]:


from collections import Counter as c
c(data["CountryCode"])


# In[226]:


c(data["CountryName"])


# In[238]:


ypred=rand.predict(x_test)
print(ypred)


# In[239]:


y_test


# In[240]:


rand.score(x_train,y_train)


# In[241]:


data.head()


# In[242]:


import pickle
pickle.dump(rand,open("co2.pickle","wb"))


# In[ ]:




