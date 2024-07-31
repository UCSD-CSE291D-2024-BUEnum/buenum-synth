import pandas as pd
import matplotlib.pyplot as plt
# Parsing the updated log data and extracting the metrics
log_data_updated = """
1,1.7949e-5,0.000147437,6,6,6,1
2,2.1023e-5,0.000129418,12,12,12,11
3,0.00025646,0.000890137,126,125,125,43
4,0.000569143,0.002337475,456,337,337,61
5,0.00283077,0.027667806,4998,1582,1582,231
6,0.065352925,0.162888318,25524,26235,3956,463
7,0.045674705,0.836265713,260430,41711,19432,3157
8,0.073710165,4.980890636,1623504,61409,39130,3791
9,0.58869307,65.379465111,15615558,229467,207188,29943
"""

# Splitting the data into lines and then columns
data_lines_updated = log_data_updated.strip().split("\n")
data_updated = [line.split(",") for line in data_lines_updated]

# Creating a DataFrame
df_updated = pd.DataFrame(data_updated, columns=["Size", "Merge_Equivs_Time", "Grow_Time", "Cache_Size", "EGraph_Size", "EGraph_Nodes", "EGraph_Classes"])
df_updated = df_updated.astype(float)  # Convert all columns to float for plotting

# Plotting size-related metrics separately
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(df_updated["Size"], df_updated["Cache_Size"], label='Cache Size', color='blue')
ax.plot(df_updated["Size"], df_updated["EGraph_Size"], label='EGraph Size', color='lightblue')
ax.plot(df_updated["Size"], df_updated["EGraph_Nodes"], label='EGraph Nodes', color='skyblue')
ax.plot(df_updated["Size"], df_updated["EGraph_Classes"], label='EGraph Classes', color='navy')

ax.set_xlabel('Size')
ax.set_ylabel('Count')
ax.set_title('Size-Related Metrics')
ax.legend()

plt.show()

# Plotting time-related metrics separately
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(df_updated["Size"], df_updated["Merge_Equivs_Time"], label='Merge Equivs Time', color='red')
ax.plot(df_updated["Size"], df_updated["Grow_Time"], label='Grow Time', color='salmon')

ax.set_xlabel('Size')
ax.set_ylabel('Time Elapsed (s)')
ax.set_title('Time-Related Metrics')
ax.legend()

plt.show()

# Adjusting plots with specified cutoffs for y-axis
# Size-related metrics with size limit
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(df_updated["Size"], df_updated["Cache_Size"], label='Cache Size', color='blue')
ax.plot(df_updated["Size"], df_updated["EGraph_Size"], label='EGraph Size', color='lightblue')
ax.plot(df_updated["Size"], df_updated["EGraph_Nodes"], label='EGraph Nodes', color='skyblue')
ax.plot(df_updated["Size"], df_updated["EGraph_Classes"], label='EGraph Classes', color='navy')

ax.set_xlabel('Size')
ax.set_ylabel('Count')
ax.set_title('Size-Related Metrics with Size Limit')
ax.legend()
ax.set_ylim(0, 1e6)

plt.show()

# Time-related metrics with time limit
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(df_updated["Size"], df_updated["Merge_Equivs_Time"], label='Merge Equivs Time', color='red')
ax.plot(df_updated["Size"], df_updated["Grow_Time"], label='Grow Time', color='salmon')

ax.set_xlabel('Size')
ax.set_ylabel('Time Elapsed (s)')
ax.set_title('Time-Related Metrics with Time Limit')
ax.legend()
ax.set_ylim(0, 5)

plt.show()
