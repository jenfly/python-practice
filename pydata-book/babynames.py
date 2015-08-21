from __future__ import division
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rc('figure', figsize=(12, 5))
np.set_printoptions(precision=4)

#!head -n 10 names/yob1880.txt

names1880 = pd.read_csv('names/yob1880.txt',
    names = ['name', 'sex', 'births'])

print(names1880)
print(names1880.groupby('sex').births.sum())

# Data ranges from 1880 to 2010
years = range(1880, 2011)

pieces = []
columns = ['name', 'sex', 'births']

for year in years:
    path = 'names/yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)

    frame['year'] = year
    pieces.append(frame)

# The variable pieces is a list of DataFrame objects, with pieces[i]
# containing the DataFrame corresponding to years[i]
print(pieces[0])
print(pieces[1])

# Concatenate everything from the list into a single DataFrame
# -- ignore_index=True means that the line number index from when the csv files
#    were read will be excluded from the output
names = pd.concat(pieces, ignore_index=True)

# Create a spreadsheet-style pivot table
total_births = names.pivot_table(values='births', index='year',
    columns='sex', aggfunc=np.sum)

print(total_births.tail())

total_births.plot(title='Total births by sex and year')

def add_prop(group):
    '''Returns the birth numbers as a proportion of the total births'''

    # Convert the number of births from int to float
    births = group.births.astype(float)

    # Divide by the total number of births
    group['prop'] = births / births.sum()
    return group

# Adds a data column 'prop' with the proportion of births, where the number
# of births is divided by the total number of births for that sex in that year
names = names.groupby(['year', 'sex']).apply(add_prop)

# Check if the sum of the 'prop' column is sufficiently close to 1 for each
# sex in each year
print(np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1))

# Play around with the groupby method to get a feel for how it works
print(names.groupby('sex').births.sum())
print(names.groupby(['sex', 'year']).births.sum())

# Get top 1000 in each group
def get_top1000(group):
    '''Returns the top 1000 data elements in a group'''
    return group.sort_index(by='births', ascending=False)[:1000]

grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(get_top1000)
print(top1000)

# Method 2 for top 1000
pieces = []
for year, group in names.groupby(['year', 'sex']):
    pieces.append(group.sort_index(by='births', ascending=False)[:1000])
top1000 = pd.concat(pieces, ignore_index=True)
top1000.index = np.arange(len(top1000))

# Analyzing naming trends
boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']
total_births = top1000.pivot_table(values='births', index='year',
    columns='name', aggfunc=np.sum)

subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots=True, figsize=(12, 10), grid=False,
    title='Number of births per year')

# Measuring the increase in naming diversity
#plt.figure()
table = top1000.pivot_table(values='prop', index='year', columns='sex',
    aggfunc=sum)
table.plot(title='Sum of table1000.prop by year and sex',
    yticks=np.linspace(0, 1.2, 13), xticks=range(1880, 2020, 10))

df = boys[boys.year == 2010]
prop_cumsum = df.sort_index(by='prop', ascending=False).prop.cumsum()
print(prop_cumsum[:10])
print(prop_cumsum.values.searchsorted(0.5))

df = boys[boys.year == 1900]
in1900 = df.sort_index(by='prop', ascending=False).prop.cumsum()
print(in1900.values.searchsorted(0.5) + 1)

def get_quantile_count(group, q=0.5):
    group = group.sort_index(by='prop', ascending=False)
    return group.prop.cumsum().values.searchsorted(q) + 1

diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')
print(diversity.head())

diversity.plot(title='Number of popular names in top 50%')

# The "Last letter" Revolution

# extract last letter from name column
get_last_letter = lambda x: x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'last_letter'
table = names.pivot_table(values='births', index=last_letters,
    columns=['sex', 'year'], aggfunc=np.sum)

# Years 1910, 1960, 2010
subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
print(subtable.head())
print(subtable.sum())
letter_prop = subtable / subtable.sum().astype(float)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female',
    legend=False)
plt.subplots_adjust(hspace=0.25)

# All years, letters d, n, y in male names
letter_prop = table / table.sum().astype(float)
dny_ts = letter_prop.ix[['d', 'n', 'y'], 'M'].T
print(dny_ts.head())
plt.close('all')
dny_ts.plot()

# Boy names that became girl names (and vice versa)
all_names = top1000.name.unique()
mask = np.array(['lesl' in x.lower() for x in all_names])
lesley_like = all_names[mask]
print(lesley_like)

filtered = top1000[top1000.name.isin(lesley_like)]
print(filtered.groupby('name').births.sum())

table = filtered.pivot_table(values='births', index='year', columns='sex',
    aggfunc=sum)
table = table.div(table.sum(1), axis=0)
table.plot(style={'M': 'k-', 'F': 'k--'})  
