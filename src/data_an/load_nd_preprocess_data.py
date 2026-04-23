import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../../data/raw/percent_bachelors_degrees_women_usa.csv")

df.info()

df = df.drop_duplicates()

# df.iloc[10:15, 0:2]


# sorting data by column name
df_sorted_agr = df.sort_values(by="Agriculture", ascending=False)


# checking highest and lowest percentage in each field
for col in df.columns[1:]:
    max_per = df[col].max()
    max_per_year = df.loc[df[col] == max_per, "Year"].iloc[0]
    print(f"Max percentage graduation for {col} is {max_per} in {max_per_year}")

    min_per = df[col].min()
    min_per_year = df.loc[df[col] == min_per, "Year"].iloc[0]
    print(f"Min percentage graduation for {col} is {min_per} in {min_per_year}")
    print()


# checking which field has highest and lowest percentage per year

df_year_index = df.set_index("Year")
for year in df["Year"].unique():
    highest_per = df_year_index.loc[year].max()
    lowest_per = df_year_index.loc[year].min()

    highest_field = df_year_index.loc[year].idxmax()
    lowest_field = df_year_index.loc[year].idxmin()

    print(f"In {year}, highest percentage is {highest_per} in {highest_field}")
    print(f"In {year}, lowest percentage is {lowest_per} in {lowest_field}")
    print()

"""
# find mean median mode

mean_df = np.mean(df["Agriculture"])
median_df = np.median(df["Agriculture"])
mode_df = stats.mode(df["Agriculture"])
"""

df.describe()


# line plot

loop_col = len(df.columns)
x_value = df.iloc[:, 0].values.tolist()
for i in range(1, loop_col):
    y_value = df.iloc[:, i].values.tolist()

    plt.plot(x_value, y_value, color="green")
    plt.xlabel("Year")
    plt.ylabel(f"{df.columns[i]}")
    plt.savefig(f"../../reports/lineplots/{df.columns[i]}_vs_year.png")
    plt.show()


# bar plot

loop_col = len(df_year_index.index.tolist())
x_bar_value = df_year_index.columns.tolist()
title_value = df_year_index.index.tolist()

for i in range(loop_col):
    y_bar_value = df_year_index.iloc[i].values.tolist()

    plt.bar(x_bar_value, y_bar_value, color="green")
    plt.xlabel("Fields")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Values")
    plt.title(title_value[i])
    plt.savefig(f"../../reports/barplots/{title_value[i]}_barplot.png")
    plt.show()
