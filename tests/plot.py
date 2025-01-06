import pandas as pd
import matplotlib.pyplot as plt


file1 = "reproduce_trajectory_with_time.csv"
file2 = "reproduce_trajectory_with_time1.csv"
file3 = "reproduce_trajectory_with_time2.csv"


data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)
data3 = pd.read_csv(file3)


data1.columns = data1.columns.str.strip()
data2.columns = data2.columns.str.strip()
data3.columns = data3.columns.str.strip()


print("File 1 Columns:", data1.columns)
print("File 2 Columns:", data2.columns)
print("File 3 Columns:", data3.columns)


fig, axes = plt.subplots(3, 1, figsize=(12, 18))


axes[0].plot(data1['x'], label='File 1 (k=6)', linestyle='-', color='blue')
axes[0].plot(data2['x'], label='File 2 (k=3)', linestyle='--', color='orange')
axes[0].plot(data3['x'], label='File 3 (k=8)', linestyle='-.', color='green')
axes[0].set_title('Comparison of x')
axes[0].set_xlabel('Index')
axes[0].set_ylabel('x Value')
axes[0].legend()
axes[0].grid()


axes[1].plot(data1['y'], label='File 1 (k=6)', linestyle='-', color='blue')
axes[1].plot(data2['y'], label='File 2 (k=3)', linestyle='--', color='orange')
axes[1].plot(data3['y'], label='File 3 (k=8)', linestyle='-.', color='green')
axes[1].set_title('Comparison of y')
axes[1].set_xlabel('Index')
axes[1].set_ylabel('y Value')
axes[1].legend()
axes[1].grid()


axes[2].plot(data1['z'], label='File 1 (k=6)', linestyle='-', color='blue')
axes[2].plot(data2['z'], label='File 2 (k=3)', linestyle='--', color='orange')
axes[2].plot(data3['z'], label='File 3 (k=8)', linestyle='-.', color='green')
axes[2].set_title('Comparison of z')
axes[2].set_xlabel('Index')
axes[2].set_ylabel('z Value')
axes[2].legend()
axes[2].grid()


plt.tight_layout()


plt.savefig("comparison_plot_k_values.png", dpi=300)


plt.show()
