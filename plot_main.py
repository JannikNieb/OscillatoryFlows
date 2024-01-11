import sys
import json
import matplotlib.pyplot as plt
from oscillatory import OscillatoryFlows

path = './data/trash'
if len(sys.argv) > 1:
    ids = sys.argv[1:]
else:
    ids = list(input("Enter id of the model to be ploted: ").split())

print(ids)

net = OscillatoryFlows(4, 'cpu', id)
fig = net.plot_results(path, ids)
plt.show()