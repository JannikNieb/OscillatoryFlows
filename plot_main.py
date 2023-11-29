import sys
import json
import matplotlib.pyplot as plt

def plot_results(path, id):
    # load data from json file
    with open(path + '/' + id + '.json', 'r', encoding='utf-8') as file:
        data_dict = json.load(file)

    cmap = plt.cm.get_cmap('inferno', int(round(len(data_dict['learned graph']) * 1.3)))
    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.suptitle(f"model {data_dict['id']}")
    ax1.plot(data_dict['loss history'])
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('training loss')

    # ax2.plot(data_dict[0], self.f(data_dict[0], self.h), label=str(i))
    for i, graph in enumerate(data_dict['learned graph']):
        ax2.plot(data_dict['samples'], graph, label=f"epoch {i * 10}", c=cmap(i))
    # ax2.legend()
    return fig


path = './data'
if len(sys.argv) > 1:
    id = sys.argv[1]
else:
    id = input("Enter id of the model to be ploted: ")

fig = plot_results(path, id)
plt.show()