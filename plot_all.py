import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data from CSV files
bot1_data = pd.read_csv('bot1_data.csv')
print(bot1_data.head()) 
bot2_data = pd.read_csv('bot2_data.csv')
print(bot2_data.head()) 
bot3_data = pd.read_csv('bot3_data.csv')
print(bot3_data.head())
# bot4_data = pd.read_csv('bot4_data.csv')
# print(bot4_data.head())  

# Merge dataframes based on the common column 'K'
print("Starting merging bot dataframes...")
combined_data = pd.merge(bot1_data, bot2_data, on='K')
# , suffixes=('_bot1', '_bot2'))
combined_data = pd.merge(combined_data, bot3_data, on='K')
# combined_data = pd.merge(combined_data, bot4_data, on='K')
print("Merging completed.")

# Plot combined data

print("Starting to plot data...")
plt.figure(figsize=(10, 6))
sns.lineplot(data=combined_data, x='K', y='Success_bot1', label='Bot 1')
sns.lineplot(data=combined_data, x='K', y='Success_bot2', label='Bot 2')
sns.lineplot(data=combined_data, x='K', y='Success_bot3', label='Bot 3')
# sns.lineplot(data=combined_data, x='K', y='Success_bot4', label='Bot 4')
plt.title('Success Rate Comparison')
plt.xlabel('Number of Aliens (K)')
plt.ylabel('Success Rate')
plt.legend()
plt.show()
print("Success Rate Plotting DOne")

plt.figure(figsize=(10, 6))
sns.lineplot(data=combined_data, x='K', y='Survival_bot1', label='Bot 1')
sns.lineplot(data=combined_data, x='K', y='Survival_bot2', label='Bot 2')
sns.lineplot(data=combined_data, x='K', y='Survival_bot3', label='Bot 3')
# sns.lineplot(data=combined_data, x='K', y='Survival Rate_bot4', label='Bot 4')
plt.title('Survival Rate Comparison')
plt.xlabel('Number of Aliens (K)')
plt.ylabel('Survival Rate')
plt.legend()
plt.show()
print("Survival Rate Plotting DOne")

print("Plotting completed.")