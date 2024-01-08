import sys
import json
import matplotlib.pyplot as plt
from oscillatory import OscillatoryFlows

path = './data/bender_data'
if len(sys.argv) > 1:
    id = sys.argv[1]
else:
    id = input("Enter id of the model to be ploted: ")

net = OscillatoryFlows(4, 'cpu', id, [1])

fig = net.plot_results(path)
plt.show()