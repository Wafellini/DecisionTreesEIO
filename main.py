import random
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

    if intr_info == 0:
        return 0.00001
    return gain(data, column) / intr_info


def treeGen(data, headers, G, prev_node, prev_choice):

    if len(set(data["Survived"])) == 1:
        return

    best = [0, 0]
    for attribute in headers[0:-1]:
        # if data.shape[0] == 1:
        #     continue
        tmp = gainRatio(data, attribute)
        if tmp > best[1]:
            best = [attribute, tmp]

    if best[0] == 0:
        return
    headers.remove(best[0])


    if prev_node is not None:
        node = prev_choice + str(best[0]) + str(random.randint(1, 1000000))
        G.add_edge(prev_node, node, weight=1, label='I')

    st = set(data[best[0]])
    for i in st:
        sub_data = data.loc[data[best[0]] == i]
        # if sub_data.shape[0] < 2:
        #     continue
        if prev_node is None:
            treeGen(sub_data, headers.copy(), G, str(best[0]), str(i))
        else:
            treeGen(sub_data, headers.copy(), G, node, str(i))



def normalizeData(data):
    headerss = list(datat.columns.values)
    headerss.remove("PassengerId")

    data["Age"] = [("Young" if i <= 20 else "Middle" if i > 20 & i <= 40 else "Old" if i > 40 else "Bruh") for i in
                   data["Age"]]

    return data, headerss


if __name__ == '__main__':
    file = "titanic-homework.csv"
    datat = pd.read_csv(file)
    columnn = 'Pclass'

    # print(entropy(data, column))
    # conditionalEntropy(data, column)
    # print(gain(data, column))
    G = nx.DiGraph()
    prev_node = None
    datat, headers = normalizeData(datat)
    # headers.remove("PassengerId")
    treeGen(datat, headers, G, prev_node, None)
    nx.draw(G, with_labels=True, font_weight='bold')

    plt.show()
