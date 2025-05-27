import pandas as pd

df1 = pd.read_csv('high_pressure_systems00_24.csv')
df2 = pd.read_csv('high_pressure_systems7000.csv')
df3 = pd.read_csv('high_pressure_systems6070.csv')
df4 = pd.read_csv('high_pressure_systems_1987112612.csv')

df_combined = pd.concat([df1, df2, df3, df4], ignore_index=True)

df_combined.to_csv('high_pressure_systems6024.csv', index=False)

