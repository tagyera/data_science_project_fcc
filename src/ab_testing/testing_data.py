import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns


# load data

df = pd.read_csv("../../data/raw/ab_test_click_data.csv")

df.head()
df.describe()

df.groupby("group").count()
df.groupby("group").sum("click")
df.groupby("group").mean("click")


# visualizing no. of clicks in controlled groups and experimental groups

palette = {0: "red", 1: "green"}

plt.figure(figsize=(10, 6))

ax = sns.countplot(x="group", hue="click", data=df, palette=palette)
plt.title("Click Distribution in Experimental and Control Groups")
plt.xlabel("Group")
plt.ylabel("Count")
plt.legend(title="Click", labels=["No", "Yes"])


group_counts = df.groupby("group").size()
group_click_counts = df.groupby(["group", "click"]).size().reset_index(name="count")


for p in ax.patches:
    height = p.get_height()
    # Find the group and click type for the current bar
    group = "exp" if p.get_x() < 0.5 else "con"
    click = 1 if p.get_x() % 1 > 0.5 else 0
    total = group_counts.loc[group]
    percentage = 100 * height / total
    ax.text(
        p.get_x() + p.get_width() / 2.0,
        height + 5,
        f"{percentage:.1f}%",
        ha="center",
        color="black",
        fontsize=10,
    )


plt.savefig("../../reports/barplots/ab_testing_plots/clicks_per_group.png")
plt.tight_layout()
plt.show()


# parameters of model from power analysis

alpha = 0.05
delta = 0.1

N_con = df[df["group"] == "con"].shape[0]
N_exp = df[df["group"] == "exp"].shape[0]

# calculating the total number of clicks per group by summing 1's

X_con = df.groupby("group")["click"].sum().loc["con"]
X_exp = df.groupby("group")["click"].sum().loc["exp"]


# computing the estimate of click probability per group

p_con_hat = X_con / N_con
p_exp_hat = X_exp / N_exp

p_pooled_hat = (X_con + X_exp) / (N_con + N_exp)


# calculating pooled variance

pooled_variance = p_pooled_hat * (1 - p_pooled_hat) * (1 / N_con + 1 / N_exp)


# computing standard error of the test

SE = np.sqrt(pooled_variance)

# computing the test statistics for z-test

test_stat = (p_con_hat - p_exp_hat) / SE

# critical value for z-test

z_crit = norm.ppf(1 - alpha / 2)


# Calculating p_values of the Z-test

# calculating p value
p_value = 2 * norm.sf(abs(test_stat))  # multiplying by 2 cause of two-tail


# function checking the statistical significance
def is_statistical_significance(p_value, alpha):

    print(f"P-value of the 2-sample Z-test: {(p_value)}")

    # Determine statistical significance
    if p_value <= alpha:
        print(
            "There is statistical significance, indicating that the observed differences between the groups are unlikely to have occurred by chance alone. This suggests that the changes in the experimental group have a real effect compared to the control group."
        )
    else:
        print(
            "There is no statistical significance, suggesting that the observed differences between the groups could have occurred by chance. This implies that the changes in the experimental group do not have a substantial effect compared to the control group."
        )


is_statistical_significance(p_value, alpha)


# Parameters for the standard normal distribution
mu = 0  # Mean
sigma = 1  # Standard deviation
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
y = norm.pdf(x, mu, sigma)


# Plotting the standard normal distribution
plt.plot(x, y, label="Standard Normal Distribution")
# Shade the rejection region for a two-tailed test
plt.fill_between(
    x,
    y,
    where=(x > z_crit) | (x < -z_crit),
    color="red",
    alpha=0.5,
    label="Rejection Region",
)
# Adding Test Statistic
plt.axvline(
    test_stat,
    color="green",
    linestyle="dashed",
    linewidth=2,
    label=f"Test Statistic = {test_stat:.2f}",
)
# Adding Z-critical values
plt.axvline(
    z_crit,
    color="blue",
    linestyle="dashed",
    linewidth=1,
    label=f"Z-critical = {z_crit:.2f}",
)
plt.axvline(-z_crit, color="blue", linestyle="dashed", linewidth=1)

# Adding labels and title
plt.xlabel("Z-value")
plt.ylabel("Probability Density")
plt.title(
    "Gaussian Distribution with Rejection Region \n (A/B Testing for LunarTech CTA button)"
)
plt.legend()
plt.savefig("../../reports/barplots/ab_testing_plots/ab_testing.png")
# Show plot
plt.show()

# Calculate the Confidence Interval (CI) for a 2-sample Z-test
## Calculate the lower and upper bounds of the confidence interval
CI = [
    round(
        (p_exp_hat - p_con_hat) - SE * z_crit, 3
    ),  # Lower bound of the CI, rounded to 3 decimal places
    round(
        (p_exp_hat - p_con_hat) + SE * z_crit, 3
    ),  # Upper bound of the CI, rounded to 3 decimal places
]

# Print the calculated confidence interval
print("Confidence Interval of the 2 sample Z-test is: ", CI)

# Here, the confidence interval provides a range of values within which the true difference between the experimental and control group proportions is likely to lie with a certain level of confidence (e.g., 95%).


def is_Practically_significant(delta, CI_95):
    """
    We assess here if the difference between Control and Experimental group is practically significant using the Minimum Detectable Effect (MDE) parameter from the Power Analysis.

    Arguments:
    - delta (float): The Minimum Detectable Effect (MDE) considered for practical significance.
    - CI_95 (tuple): A tuple representing the lower and upper bounds of the 95% Confidence Interval.

    Returns:
    - Prints whether the experiment has practical significance based on the MDE and CI.
    """

    # Extract lower bound of 95% Confidence Interval
    lower_bound_CI = CI_95[0]

    # Check if the lower bound of the CI is greater than or equal to delta
    if delta <= lower_bound_CI:
        print(
            f"We have practical significance! \nWith MDE of {delta}, The difference between Control and Experimental group is practically significant."
        )
        return True
    else:
        print(
            "We don't have practical significance! \nThe difference between Control and Experimental group is not practically significant."
        )
        return False


# Call the function
significance = is_Practically_significant(delta, CI)
print(f"Lower bound of 95% confidence interval is: {CI[0]}")
