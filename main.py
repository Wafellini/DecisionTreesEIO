import random
from math import log2

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import treeGraph


def entropy(data, column):
    lst = list(data[column])
    st = set(lst)

    entrpy = 0
    for i in st:
        entrpy -= log2(lst.count(i) / len(lst)) * lst.count(i) / len(lst)

    print("Entropy - " + f": {entrpy}")
    return entrpy


def conditionalEntropy(data, column):
    print("Conditional Entropy - " + column)
    st = set(data[column])

    con_entr = 0
    for i in st:
        print(f"   {i}")
        classs = data.loc[data[column] == i]
        con_entr += entropy(classs, "Survived") * classs.shape[0] / data.shape[0]

    print("Conditional Entropy - " + column+ f": {con_entr}")
    return con_entr


def gain(data, column):
    print("Gain for " + column)
    gin = entropy(data, "Survived") - conditionalEntropy(data, column)
    print("Gain - " + column + f": {gin}")
    return gin


def gainRatio(data, column):
    print('===================================')
    print("Gain Ratio for " + column)
    st = set(data[column])

    intr_info = 0
    for i in st:
        lst = data.loc[data[column] == i]
        intr_info -= log2(lst.shape[0] / data.shape[0]) * lst.shape[0] / data.shape[0]

    if intr_info == 0:
        return 0.00001
    gr = gain(data, column) / intr_info
    print("Gain Ratio - " + column + f": {gr}")
    return gr


def treeGen(data, headers, G, prev_node, prev_choice):
    if len(set(data["Survived"])) == 1:
        return

    best = [0, 0]
    for attribute in headers[0:-1]:
        tmp = gainRatio(data, attribute)
        if tmp > best[1]:
            best = [attribute, tmp]


    if best[0] == 0:

        G.add_edge(prev_node, "Prev choice: " + prev_choice + "\nSurvived: " +
                   str(list(data["Survived"])[0]) + "\n " + str(
            random.randint(1, 1000000)), weight=1,
                   label=prev_choice)
        return
    headers.remove(best[0])

    print("||||||||||||||||||||||||||||||||||||||||||||||")
    print("||||||||||||||||||||||||||||||||||||||||||||||")
    print("Chosen attribute - ", best[0])
    print("||||||||||||||||||||||||||||||||||||||||||||||")
    print("||||||||||||||||||||||||||||||||||||||||||||||")

    if prev_node is not None:
        node = "Prev choice: " + prev_choice + "\nVal: " + str(best[0]) + "\n" + str(random.randint(1, 1000000))
        G.add_edge(prev_node, node, weight=1, label=prev_choice)

    st = set(data[best[0]])
    for i in st:
        sub_data = data.loc[data[best[0]] == i]
        if len(set(sub_data["Survived"])) == 1:
            G.add_edge(node, "Prev choice: " + str(i) + "\nSurvived: " +
                       str(list(sub_data["Survived"])[0]) + "\n " + str(
                random.randint(1, 1000000)), weight=1,
                       label=prev_choice)
            continue
        if prev_node is None:
            treeGen(sub_data, headers.copy(), G, str(best[0]), str(i))
        else:
            treeGen(sub_data, headers.copy(), G, node, str(i))


def normalizeData(data):
    headerss = list(datat.columns.values)
    headerss.remove("PassengerId")

    data["Age"] = [("Young" if i <= 20 else "Middle" if i > 20 & i <= 40 else "Old" if i > 40 else "Error") for i in
                   data["Age"]]

    return data, headerss


if __name__ == '__main__':
    file = "titanic-homework.csv"
    datat = pd.read_csv(file)
    columnn = 'Pclass'

    G = nx.DiGraph()
    prev_node = None
    datat, headers = normalizeData(datat)

    treeGen(datat, headers, G, prev_node, None)

    pos = treeGraph.hierarchy_pos(G, "Sex")
    nx.draw(G, pos=pos, with_labels=True, font_size=9)
    plt.show()
