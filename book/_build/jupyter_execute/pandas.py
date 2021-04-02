# Pandas

Pandas is a library for tabular data (dataframe).

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(100, 2), columns=['x', 'y'])
df.head()

df.shape

df.info()

df.describe()

dates = pd.date_range("20210101", periods=100)
dates[0:5]

df = pd.DataFrame(np.random.randn(100, 2), columns=['x', 'y'], index=dates)
df.head()

df.plot();

df['2021-02-11':'2021-02-15']

df.loc[df.x < 0.5]

df['label'] = [chr(97 + int(num)) for num in abs(df.x.values) * 10]

df.head()

df.loc[df.label == 'a']

df.loc[df.label == 'a'].plot('x', 'y', kind='scatter');

df.groupby(by=df["label"])

df.groupby(by=df["label"]).sum()

df1 = pd.DataFrame(np.random.randn(5, 2), columns=['x', 'y'])
df2 = pd.DataFrame(np.random.randn(5, 2), columns=['x', 'y'])

df1

df2

pd.concat([df1, df2], ignore_index=True)

df_titanic = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
df_titanic.head()

df_titanic_grouped = df_titanic.groupby('Embarked')

(df_titanic_grouped.sum() / df_titanic_grouped.count()).plot.bar(
    y='Survived',
    ylabel='Passengers that survived per embarkment\n(%)',
    xlabel='Port of Embarkation\n(C = Cherbourg; Q = Queenstown; S = Southampton)'
);

pd.read_csv?

For more information, see the [documentation](https://pandas.pydata.org/).

