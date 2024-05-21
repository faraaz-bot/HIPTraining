import matplotlib.pyplot as plt
import numpy as np

x_data=[]
y_data=[]
with open("../build/without_hipgraph_gfx942_1000000.data", "r") as file1:
    lines = file1.readlines();

    print(lines)

    x_data = [int(i) for i in lines[0].split(",")]
    y_data = [float(i) for i in lines[1].split(",")]

for i in x_data:
    print(i)

for i in y_data:
    print(i)

x_hipgraph_data=[]
y_hipgraph_data=[]
with open("../build/with_hipgraph_gfx942_1000000.data", "r") as file1:
    lines = file1.readlines();

    print(lines)

    x_hipgraph_data = [int(i) for i in lines[0].split(",")]
    y_hipgraph_data = [float(i) for i in lines[1].split(",")]

for i in x_hipgraph_data:
    print(i)

for i in y_hipgraph_data:
    print(i)

plt.plot(x_data, y_data, 'k--', x_hipgraph_data, y_hipgraph_data, 'g--')
plt.title("Performance of hipGraph gfx942 rocm-6.2 (m = 1000000)");
plt.xlabel("Iterations");
plt.ylabel("Wall time (seconds)")
plt.legend(["Without hipGraph", "With hipGraph"])
plt.show()
plt.savefig('../build/hipgraph_perf_gfx942_1000000.png')
