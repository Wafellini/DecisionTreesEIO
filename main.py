from math import log

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def entropy(data, column):
    lst = list(data[column])
    st = set(lst)

    entrpy = 0
    for i in st:
        entrpy -= log(lst.count(i) / len(lst)) * lst.count(i) / len(lst)

    return entrpy


def conditionalEntropy(data, column):
    st = set(data[column])

    con_entr = 0
    for i in st:
        bruh = data.loc[data[column] == i]
        con_entr += entropy(bruh, "Survived") * bruh.shape[0] / data.shape[0]

    return con_entr


def gain(data, column):
    return entropy(data, "Survived") - conditionalEntropy(data, column)


def gainRatio(data, column):
    st = set(data[column])

    intr_info = 0
    for i in st:
        lst = data.loc[data[column] == i]
        intr_info -= log(lst.shape[0] / data.shape[0]) * lst.shape[0] / data.shape[0]

    return gain(data, column) / intr_info


if __name__ == '__main__':
    file = "titanic-homework.csv"
    data = pd.read_csv(file)
    column = 'Pclass'

    # print(entropy(data, column))
    conditionalEntropy(data, column)
    print(gain(data, column))

def treeGen():
    G = nx.Graph()
    G.add_edge(1, 2)  # default edge data=1
    G.add_edge(2, 3, weight=0.1)  # specify edge data


    nx.draw(G, with_labels=True, font_weight='bold')


    plt.show()
treeGen()