import matplotlib.pyplot as plt
import numpy as np

# Data
growing_to_size = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
ecache_size_first = np.array([6, 7, 11, 18, 28, 41, 57, 76, 98])
cache_size_second = np.array([6, 12, 126, 456, 4998, 25524, 260430, 1623504, 15615558])
egraph_size = np.array([6, 12, 125, 337, 1582, 26235, 41711, 61409, 229467])
egraph_nodes = np.array([6, 12, 125, 337, 1582, 3956, 19432, 39130, 207188])
egraph_classes = np.array([1, 11, 43, 61, 231, 463, 3157, 3791, 29943])
merge_equivs_time_first = np.array([3.3827e-5, 2.8586e-5, 0.000244899, 0.000695556, 0.002808853, 0.078407748, 0.057941086, 0.086426265, 0.835069856])
merge_equivs_time_second = np.array([1.7949e-5, 2.1023e-5, 0.00025646, 0.000569143, 0.00283077, 0.065352925, 0.045674705, 0.073710165, 0.58869307])
grow_time_first = np.array([0.000420185, 0.000199259, 0.001306544, 0.003721421, 0.044911251, 0.219771299, 1.220129593, 11.178083978, 121.517373545])
grow_time_second = np.array([0.000147437, 0.000129418, 0.000890137, 0.002337475, 0.027667806, 0.162888318, 0.836265713, 4.980890636, 65.379465111])

# Plot setup
# Create two subplots side by side
fig1, axs1 = plt.subplots(1, 2, figsize=(15, 6))

# Figure 1: Size related metrics (original scale)
axs1[0].plot(growing_to_size, ecache_size_first, label='ECache Size', marker='o')
axs1[0].plot(growing_to_size, cache_size_second, label='Cache Size', marker='x')
axs1[0].plot(growing_to_size, egraph_size, label='EGraph Size', linestyle='--')
axs1[0].plot(growing_to_size, egraph_nodes, label='EGraph Nodes', linestyle='--')
axs1[0].plot(growing_to_size, egraph_classes, label='EGraph Classes', linestyle='--')
axs1[0].set_xlabel('Growing to Size')
axs1[0].set_ylabel('Size Metrics')
axs1[0].set_title('Figure 1.1: Size Related Metrics Comparison (Original Scale)')
axs1[0].legend()

# Figure 1: Size related metrics (log scale)
axs1[1].plot(growing_to_size, ecache_size_first, label='ECache Size', marker='o')
axs1[1].plot(growing_to_size, cache_size_second, label='Cache Size', marker='x')
axs1[1].plot(growing_to_size, egraph_size, label='EGraph Size', linestyle='--')
axs1[1].plot(growing_to_size, egraph_nodes, label='EGraph Nodes', linestyle='--')
axs1[1].plot(growing_to_size, egraph_classes, label='EGraph Classes', linestyle='--')
axs1[1].set_yscale('log')
axs1[1].set_xlabel('Growing to Size')
axs1[1].set_ylabel('Size Metrics (Log Scale)')
axs1[1].set_title('Figure 1.2: Size Related Metrics Comparison (Log Scale)')
axs1[1].legend()

plt.tight_layout()
plt.show()

# Create a new figure for Figure 2 and Figure 3
fig2, axs2 = plt.subplots(2, 2, figsize=(15, 12))

# Figure 2: Time Elapsed for merge_equivs (original scale)
axs2[0, 0].plot(growing_to_size, merge_equivs_time_first, label='merge_equivs Time ECache', marker='o')
axs2[0, 0].plot(growing_to_size, merge_equivs_time_second, label='merge_equivs Time Cache', marker='x')
axs2[0, 0].set_xlabel('Growing to Size')
axs2[0, 0].set_ylabel('Time Elapsed (s)')
axs2[0, 0].set_title('Figure 2.1: Time Elapsed for merge_equivs Comparison (Original Scale)')
axs2[0, 0].legend()

# Figure 2: Time Elapsed for merge_equivs (log scale)
axs2[0, 1].plot(growing_to_size, merge_equivs_time_first, label='merge_equivs Time ECache', marker='o')
axs2[0, 1].plot(growing_to_size, merge_equivs_time_second, label='merge_equivs Time Cache', marker='x')
axs2[0, 1].set_yscale('log')
axs2[0, 1].set_xlabel('Growing to Size')
axs2[0, 1].set_ylabel('Time Elapsed (s) (Log Scale)')
axs2[0, 1].set_title('Figure 2.2: Time Elapsed for merge_equivs Comparison (Log Scale)')
axs2[0, 1].legend()

# Figure 3: Time Elapsed for grow (original scale)
axs2[1, 0].plot(growing_to_size, grow_time_first, label='grow Time ECache', marker='o')
axs2[1, 0].plot(growing_to_size, grow_time_second, label='grow Time Cache', marker='x')
axs2[1, 0].set_xlabel('Growing to Size')
axs2[1, 0].set_ylabel('Time Elapsed (s)')
axs2[1, 0].set_title('Figure 3.1: Time Elapsed for grow Comparison (Original Scale)')
axs2[1, 0].legend()

# Figure 3: Time Elapsed for grow (log scale)
axs2[1, 1].plot(growing_to_size, grow_time_first, label='grow Time ECache', marker='o')
axs2[1, 1].plot(growing_to_size, grow_time_second, label='grow Time Cache', marker='x')
axs2[1, 1].set_yscale('log')
axs2[1, 1].set_xlabel('Growing to Size')
axs2[1, 1].set_ylabel('Time Elapsed (s) (Log Scale)')
axs2[1, 1].set_title('Figure 3.2: Time Elapsed for grow Comparison (Log Scale)')
axs2[1, 1].legend()

plt.tight_layout()
plt.show()
