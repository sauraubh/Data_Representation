#!/usr/bin/env python
# coding: utf-8

# # Project

# Goal of the Project: Market Research to to open a small robot-run cafe in Los Angeles

# In[1]:


get_ipython().system('pip install usaddress')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
import usaddress
import re


# Imported all required liabraries

# <div class="alert alert-success" role="alert">
# Reviewer's comment v. 1:
#     
# Well done that you formulated a goal of the project.
# </div>

# ## Import Data 

# In[3]:


data = pd.read_csv('/datasets/rest_data_us.csv', sep=',')

data.info()
data.head()


# opened data file and looked for general information and displayed table contents which are as follows:
# •	object_name — establishment name
# •	chain — chain establishment (TRUE/FALSE)
# •	object_type — establishment type
# •	address — address
# •	number — number of seats
# 

# ## Pre-process Data 

# In[4]:


percentage = data.duplicated(keep=False).value_counts(normalize=True) * 100
print (percentage)


# In[5]:


data.duplicated(keep=False).mean()


# In[6]:


data = data.drop_duplicates()


# We Don't have any duplicates so may be we don't need to drop them

# Dropped duplicates If we have any

# In[7]:


data.columns = ['id', 'name', 'address', 'chain', 'type', 'total_seats']


# Changed column names to smooth run coding process further

# In[8]:


data['chain'].dropna(inplace=True)


# Dropped the empty values we have in coulmn chain

# In[9]:


data.replace("", float("NaN"), inplace=True)
data.dropna(subset = ["name"], inplace=True)


# Replaced empty column name with NaN and then dropped it

# In[10]:


def cleanNames(name):
    name = name

    # if # symbol in name, split it to remove numbers
    if '#' in name:
        name = name.split('#')[0].strip()
    
    # if - symbol in name, split it to remove numbers
    if '-' in name:
        tmp = name.split('-')

        if tmp[1].isdecimal():
            name = tmp[0].strip()
    
    return name
data['name'] = data.name.apply(cleanNames)
data.head()


# Defined clean names to apply to name column in our data as lot of names were having special characters and unnecessary symbols and space so using conditional argument with split function we have cleared the name column to get clarity on data

# In[11]:


def cleanAddress(address):
    address = address

    # hardcode singular cases
    if address.startswith('OLVERA'):
        address = 'OLVERA,Los Angeles,USA'
    elif address.startswith('1033 1/2 LOS ANGELES ST'):
        address = '1033 1/2 LOS ANGELES ST,Los Angeles, USA'
        
    # standard cases
    else:
        raw = usaddress.parse(address)
        addressDict = {}
        for i in raw:
            addressDict.update({i[1]:i[0]})
        
        if 'StreetNamePostType' in addressDict:
            address = addressDict['AddressNumber'] + " " + str(addressDict['StreetName']) +                 " " + str(addressDict['StreetNamePostType'])+str(',Los Angeles,USA')
        else:
            address = addressDict['AddressNumber'] + " " + str(addressDict['StreetName']) +                 " "+str(',Los Angeles,USA')
    
    return address
data['address'] = data.address.apply(cleanAddress)
data.head()


# Here we used same method to get clear address as here in address column some of the rows have one word or singular address which we changed to full address by adding city name and state following to it.
# we created address dictionary type to get two types of address, the address with single word and address starts with street number. we have set up format for each type. and parsed it following to,
# When its one word address we used One word + state + country and for other we used address number+ street name + city + contry
# 

# ## Investigate the proportions of the various types of establishments 

# In[12]:


typeData = data['type'].value_counts().rename_axis('type').reset_index(name='count')
typeData


# to get which establishments have how much share in this area we used value counts and then we are gonna set up pychart for clear understanding

# In[13]:


plt.figure(figsize=(10, 5))
plt.pie(typeData['count'], labels=typeData['type'],autopct='%0.f%%', shadow=True, startangle=145)
plt.title('Types of Establishments')
plt.show()


# The breakdown of the various types of establishments can be seen in the pie chart above: 75% of establishments are restaurants, 11% of establishments are fast food, and the rest are broken down between cafe, pizza, bar and bakery.

# It depends on the density of population and tourists, Considering both the aspects LA is one of the densed place for tourist so, that is the reason city may have to offer different kinds of option to them such as fine-dinning, Healthy food etc, on the other hand some may like to just eat different type of food when they are on trip and LA provides you that experiance

# ## Investigate the proportions of chain and nonchain establishments 

# In[14]:


chainData = data['chain'].value_counts().rename_axis('chain').reset_index(name='count')
chainData


# Here we did the same thing as above we are gonna count number of share per establishment chain or no-chain

# In[15]:


plt.pie(chainData['count'], labels=chainData['chain'],autopct='%0.f%%', shadow=True, startangle=145);
plt.title('Chain vs Non-Chain Establishments');


# The breakdown between chain and nonchain establishments can be seen in the pie chart above. More than half (62%) of the establishments are nonchain estabilshments, while 38% of the establishments belong to a chain.

# ## Which type of establishment is typically a chain? 

# In[16]:


typeChainData = data[data['chain'] == True]
typeChainData.head()


# we have created new dataframe here to only get chain data from the rest of the data so if chain coulumn is true we are gonna drag that to this dataframe

# In[17]:


typeChainData = pd.pivot_table(typeChainData,index=['chain','type'], values=['id'], aggfunc='count').reset_index()
typeChainData.columns = ['chain', 'type', 'count']
typeChainData = typeChainData.sort_values(by='count', ascending=False)
typeChainData


# Here we have created pivot table to get number of share per establishment in chain type and then we sorted it by descending order

# In[18]:


totalData = data['type'].value_counts().rename_axis('type').reset_index(name='total')
totalData


# Here we have calculated and sorted the share of total establishment in our old data

# In[19]:


typeChainData = pd.merge(typeChainData, totalData, how='inner', on='type')
typeChainData['ratio'] = typeChainData['count'] / typeChainData['total'] * 100
typeChainData


# here we have merged our both DF inner and we have calculated the share ratio percent by dividing chain establishment by total establishment, count of both

# In[20]:


plt.figure(figsize=(10, 7))

ax = sns.barplot(data = typeChainData.sort_values('type', ascending=False), 
                 x='type', 
                 y='ratio')
plt.xlabel('Type of Establishment')
plt.ylabel('Ratio of Chains')
plt.title('Ratio of Chains per Establishment')
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                xy=(p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', 
                va = 'center', 
                xytext = (0, 10), 
                textcoords = 'offset points')
plt.show()


# The breakdown of which types of establishments are typically chains can be seen in the bar chart above. As it shows, all bakeries are part of a chain as it seems that no one can handle that business on their own, more than half of cafes and fast food establishments are chains and less than half of pizza places, restaurants and bars are chains.

# ## What characterizes chains? 

# In[21]:


chainData = data[data['chain'] == True]
chainData = pd.pivot_table(chainData, index=['total_seats'], values=['id'], aggfunc=['count']).reset_index()
chainData.columns = ['total_seats','count']
chainData = chainData.sort_values(by='count', ascending=False)
chainData.head(10)


# here again we have took the chains separate from our data to calculate the number of seats each establishment can have in chain type to feed its customer

# In[22]:


nonChainData = data[data['chain'] == False]
nonChainData = pd.pivot_table(nonChainData, index=['total_seats'], values=['id'], aggfunc=['count']).reset_index()
nonChainData.columns = ['total_seats','count']
nonChainData = nonChainData.sort_values(by='count', ascending=False)
nonChainData.head(10)


# here we did the same as above and counted who has more capacity to calm their customers hunger by showing them in descending order

# In[23]:


plt.scatter(chainData['total_seats'], chainData['count'], color='r', label='chains')
plt.scatter(nonChainData['total_seats'], nonChainData['count'], color='g', label='non-chains')
plt.xlabel('Total Seats per Establishment')
plt.ylabel('Number of Establishments')
plt.legend()
plt.title('Correlation between number of seats and number of establishments')
plt.show()


# The scatter plot above shows the correlation between the number of seats and the number of establishments. In general, most of the establishments have 50 or less seats, while less of the establishments have 50 or more seats. However, the nonchain establishments typically have more seats than the chain establishments. This can be seen in both the areas: more nonchain restaurants than chain restaurants have less than 50 seats and more nonchain restaurants than chain restaurants have more than 50 seats. Chains seem to be characterized by many establishments with a small number of seats.

# In[24]:


import plotly.express as px
df = px.data.tips()
df = chainData[['total_seats','count']]
df2 = nonChainData[['total_seats','count']]

fig = px.line(df, x="total_seats", y="count", text="total_seats", log_x=True, color="count")
fig.update_traces(mode="markers+lines",hovertemplate=None)
fig.update_layout(title_text='Chain', title_x=0.5)
fig.show()
fig = px.line(df2, x="total_seats", y="count", text="total_seats", log_x=True,color="count")
fig.update_traces(mode="markers+lines",hovertemplate=None)
fig.update_layout(title_text='Nonchain', title_x=0.5)
fig.show()



# Here we have plotly express graph where our investors can have better understanding of whats going in as scatter plot above will not impress them this may be as they can see each result line by line by clicking and playing with hover in the side

# ## Determine the average number of seats for each type of establishment.

# In[25]:


labels = ['Cafe', 'Restaurant', 'Fast Food', 'Pizza', 'Bar', 'Bakery']    
values = []


# bar graph labels and values

# In[26]:


for establishment in range(len(labels)):
    estType = labels[establishment]
    currData = data[data['type'] == estType]
    avgSeats = currData['total_seats'].mean()
    values.append(avgSeats)


# here we are calculating avarage number of seats per establishment 

# In[27]:


df = pd.DataFrame({"Establishment":labels, "Seats":values})

plt.figure(figsize=(10, 7))

ax = sns.barplot(data = df.sort_values('Seats', ascending=False), 
                 x='Establishment', 
                 y='Seats')
plt.xlabel('Type of Establishment')
plt.ylabel('Average Number of Seats')
plt.title('Average Number of Seats per Establishment')
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                xy=(p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', 
                va = 'center', 
                xytext = (0, 10), 
                textcoords = 'offset points')


plt.show()


# here we used declared label above to get clear bar graph and also we used seats avarage as our values.The bar graph above shows the average number of seats for each type of establishment. On average, restaurants have the greatest amount of seats around 50 which makes sense as customer tends to eat heavy and get drunk in the dinnertime specially.

# ## Put the data on street names from the address column in a separate column 

# In[28]:


def streetNames(street):
    street = street.split(',')[0].split(' ')[-2:]
    street = " ".join(street)
    if street == '103 ':
        street = '103RD ST'
    return street
data['street'] = data.address.apply(streetNames)
data


# here we are using code to separate street name from address which we mixed in the first step and we are creating new column to dataset

# ## Plot a graph of the top ten streets by number of establishments 

# In[29]:


streetData = pd.pivot_table(data, index=['street'], values=['id'], aggfunc=['count']).reset_index()
streetData.columns = ['street','count']
topStreetData = streetData.sort_values(by='count', ascending=False).head(10)
topStreetData


# Here we have created a pivot table with the help of street column which we invented in above step and now we are calculating the avarage establishment or which street has more number establishments by counting and presenting it in descending order.

# In[30]:


plt.figure(figsize=(10, 7))
ax = sns.barplot(data = topStreetData.sort_values('count', ascending=False), 
                 x='street', 
                 y='count')
plt.xlabel('Most Popular Streets')
plt.ylabel('Number of Establishments')
plt.title('Number of Establishments per Most Popular Streets')
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                xy=(p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', 
                va = 'center', 
                xytext = (0, 10), 
                textcoords = 'offset points')

plt.xticks(rotation=45)
plt.show()


# The bar chart above shows the top ten streets by the number of establishments located on the street. The most popular streets for establishments where custumor wants feed its hunger are Sunset Boulevard, Wilshire Boulevard, Pico Boulevard and Western Avenue, God bless those streets.

# It is not possible that each corner of city will be filled with restaurants there are other aspects as I explained in my final conclusion, These streets have more number of footnote(Customer number) as well as they might be in prime location at the city that is why there is difference in density of restaurant which may vary depending on the area and its facilities

# ## Find the number of streets that only have one restaurant 

# In[31]:


singleStreetData = streetData[streetData['count'] == 1]
numStreets = len(singleStreetData)
print(numStreets)


# here we are taking the number of streets who has only one restaurant in it and There are 237 with only one restaurant on them, may be those are bad streets down there

# It is not possible that each corner of city will be filled with restaurants there are other aspects as I explained in my final conclusion, These streets have more number of footnote(Customer number) as well as they might be in prime location at the city that is why there is difference in density of restaurant which may vary depending on the area and its facilities

# ## For streets with a lot of establishments, look at the distribution of the number of seats 

# In[32]:


avgNumRestaurants = streetData['count'].mean().round()
avgStreetData = streetData[streetData['count'] > avgNumRestaurants]


# Here we have clculated avarage number of restaurants per street and then we have extracted those streets who have more number of restaurant than avarage as we don't want anything below avarage here so that we can convince our investor with strong points which we have

# In[33]:


ax = sns.distplot(avgStreetData['count'],rug_kws={"color": "g"},
                  kde_kws={"color": "k", "lw": 3, "label": "count"},
                  hist_kws={"histtype": "step", "linewidth": 3,
                            "alpha": 1, "color": "g"})
ax.set_title('Distribution of the Number of Seats')
ax.set_xlabel('Number of Seats')
ax.set_ylabel('Density of Streets')
plt.show()


# For streets with a lot of establishments (greater than the average number of establishments per street), most of those establishments have 100 or less seats. In fact, it seems that many of those establishments have less than 50 seats.

# ## conclusion and recommendations  

# we have decided to open a small robot-run cafe in Los Angeles. The project is promising but expensive, so we need investor to give us more money. The investors are interested in the current market conditions in order to determine if we will be able to maintain your success when the novelty of robot waiters wears off without having any apocolyptic effect on city of Angels.
# 
# After an analysis of current food establishment market in Los Angeles, there are several conclusions that can be made regarding the data. 
# 
# 1. Most Popular Establishment: Restaurant or Bar, Restaurant and Bar
# 2. Avarage number of seat per Restaurant is 48 and Bar is 45 so, we can have capacity to offer more with the help of future human nemesis(robots)
# 3. The Sincity(LA) has some streets with more than 250 establishments on them which ensures footnote(customers number is more) those streets are: Sunset Boulevard, Wilshire Boulevard, Pico Boulevard, Western Avenue, Figueroa Street, Olympic Boulevard, Vermount Avenue, Monica Boulevard, 3rd Street and Holly Boulevard
# 4. The establishments on these streets have average 20-100 seats
# 
# We have decided to go with Nonchain Restaurant & Bar with 50-60 sits on one of those streets where we can help our customers to have good start at LA to enjoy there journey along.

# ## Prsentation Link 

# Presentation: <https://drive.google.com/file/d/1o29JHHpr03-hJdh20ZMfL4sHaPq8VYjS/view?usp=sharing>
