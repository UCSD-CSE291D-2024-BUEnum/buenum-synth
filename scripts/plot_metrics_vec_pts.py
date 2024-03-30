import matplotlib.pyplot as plt
import numpy as np

# Data for merge_equivs times across all runs
merge_equivs_times = np.array([
    [2.2429e-5, 3.5356e-5, 9.0065e-5, 0.000186698, 0.001766502, 0.055151557, 0.042342092, 0.067249014, 0.614532442],
    [1.3144e-5, 1.8457e-5, 8.648e-5, 0.000194353, 0.001650673, 0.063444483, 0.047908508, 0.065530541, 0.586319406],
    [1.6747e-5, 2.6236e-5, 0.000271783, 0.000413362, 0.002512268, 0.056877509, 0.043094261, 0.066972822, 0.590183196],
    [1.6245e-5, 2.5657e-5, 0.000207068, 0.000580906, 0.002649983, 0.055295948, 0.041166028, 0.065437856, 0.579128602],
    [1.2207e-5, 1.8694e-5, 0.000276492, 0.000561074, 0.003198972, 0.055859864, 0.042252186, 0.070794263, 0.586458569]
])

# Grow times from all 5 runs
grow_times = np.array([
    [0.00016902, 0.000154976, 0.001108798, 0.001476534, 0.019691098, 0.188963602, 1.221265748, 10.344168807, 116.178968739],
    [9.8085e-5, 8.62e-5, 0.000668561, 0.001417659, 0.019330586, 0.201232955, 1.274587923, 10.562930885, 116.301577501],
    [0.000123366, 0.00016307, 0.001368755, 0.003098469, 0.041843113, 0.189792796, 1.231662835, 10.547843797, 116.991712372],
    [0.000133932, 0.000110018, 0.001313244, 0.003664965, 0.042997361, 0.185676294, 1.211632691, 10.755461689, 116.059588051],
    [0.0001063, 8.5026e-5, 0.001524957, 0.003724542, 0.04326678, 0.190838301, 1.215392388, 10.89965607, 116.809427553]
])

# Data for merge_equivs times for IOPairMap across all 5 runs
merge_equivs_iopairmap = np.array([
    [1.3474e-5, 2.1433e-5, 0.000240921, 0.000247484, 0.001751445, 0.073078869, 0.052423524, 0.080998096, 0.701779283],  # Run 1
    [1.7229e-5, 2.4373e-5, 0.000259768, 0.000588942, 0.001629415, 0.074030989, 0.052940133, 0.083036926, 0.698950261],  # Run 2
    [2.1596e-5, 2.9777e-5, 0.000329716, 0.000762467, 0.001783267, 0.072941807, 0.051328162, 0.081801345, 0.706315608],  # Run 3
    [1.7884e-5, 2.8153e-5, 0.00035544, 0.00074616, 0.001475763, 0.07383526, 0.06054136, 0.080998096, 0.700620759],  # Run 4
    [1.6246e-5, 2.4336e-5, 0.000344981, 0.000688745, 0.001789962, 0.071561612, 0.049116027, 0.078546902, 0.693453701]  # Run 5
])

# Data for grow times for IOPairMap across all 5 runs
grow_times_iopairmap = np.array([
    [0.00017076, 0.000149188, 0.00118164, 0.002501855, 0.019062468, 0.214024927, 1.18201885, 10.257843252, 115.770524957],  # Run 1
    [0.000208711, 0.000175653, 0.001351087, 0.003280611, 0.037539861, 0.223579043, 1.23394775, 10.384504529, 115.293171599],  # Run 2
    [0.000147308, 0.000180649, 0.001072715, 0.00299422, 0.020495602, 0.216803881, 1.190975215, 10.169704169, 114.794991927],  # Run 3
    [0.000219566, 0.000201053, 0.001685059, 0.004084213, 0.025601986, 0.217070683, 1.215704371, 10.89965607, 116.809427553],  # Run 4
    [0.000199949, 0.000160573, 0.001560143, 0.003779632, 0.022759204, 0.209866314, 1.204232845, 10.362883843, 115.512711863]   # Run 5
])

# Calculate max, min, and mean
max_times = merge_equivs_times.max(axis=0)
min_times = merge_equivs_times.min(axis=0)
mean_times = merge_equivs_times.mean(axis=0)
growing_to_size = np.arange(1, 10)

# Calculate max, min, and mean for IOPairMap
max_times_iopairmap = merge_equivs_iopairmap.max(axis=0)
min_times_iopairmap = merge_equivs_iopairmap.min(axis=0)
mean_times_iopairmap = merge_equivs_iopairmap.mean(axis=0)
sizes_iopairmap = np.arange(1, 10)

# Plotting
plt.figure(figsize=(10, 6))

# Min and Max as light shadow
plt.fill_between(growing_to_size, min_times, max_times, color='skyblue', alpha=0.5, label='Min-Max Range for IOPairVec')

# Mean as deeper line
plt.plot(growing_to_size, mean_times, color='blue', marker='o', linestyle='-', linewidth=2, markersize=5, label='Mean Time for IOPairVec')

# Min and Max as light shadow for IOPairMap
plt.fill_between(sizes_iopairmap, min_times_iopairmap, max_times_iopairmap, color='lightcoral', alpha=0.3, label='Min-Max Range for IOPairMap')

# Mean as deeper line for IOPairMap
plt.plot(sizes_iopairmap, mean_times_iopairmap, color='red', marker='o', linestyle='-', linewidth=2, markersize=5, label='Mean Time for IOPairMap')

plt.title('Time Elapsed for merge_equivs Across Runs')
plt.xlabel('Growing to Size')
plt.ylabel('Time Elapsed for merge_equivs (s)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate max, min, and mean for grow times
max_grow_times = grow_times.max(axis=0)
min_grow_times = grow_times.min(axis=0)
mean_grow_times = grow_times.mean(axis=0)
sizes = np.arange(1, 10)

# Min and Max as light shadow for variability range
plt.fill_between(sizes, min_grow_times, max_grow_times, color='lightgreen', alpha=0.3, label='Min-Max Range for IOPairVec')

# Mean as deeper line
plt.plot(sizes, mean_grow_times, color='green', marker='o', linestyle='-', linewidth=2, markersize=5, label='Mean Grow Time (IOPairVec)')

# Calculate max, min, and mean for grow times related to IOPairMap
max_grow_times_iopairmap = grow_times_iopairmap.max(axis=0)
min_grow_times_iopairmap = grow_times_iopairmap.min(axis=0)
mean_grow_times_iopairmap = grow_times_iopairmap.mean(axis=0)
sizes_grow_iopairmap = np.arange(1, 10)

# Min and Max as light shadow for variability range
plt.fill_between(sizes_grow_iopairmap, min_grow_times_iopairmap, max_grow_times_iopairmap, color='yellow', alpha=0.3, label='Min-Max Range for IOPairMap')

# Mean as deeper line
plt.plot(sizes_grow_iopairmap, mean_grow_times_iopairmap, color='orange', marker='o', linestyle='-', linewidth=2, markersize=5, label='Mean Grow Time (IOPairMap)')

plt.title('Grow Time Cost Across Runs')
plt.xlabel('Growing to Size')
plt.ylabel('Time Elapsed for grow (s)')
plt.legend()
plt.grid(True)
plt.show()
