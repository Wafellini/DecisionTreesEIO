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


def treeGen(data, headers, G, GG):
    while len(headers) > 1:
        if len(set(data["Survived"])) == 1:
            break

        best = [0, 0]
        for attribute in headers[0:-1]:
            # if data.shape[0] == 1:
            #     continue
            tmp = gainRatio(data, attribute)
            if tmp > best[1]:
                best = [attribute, tmp]

        if best[0] == 0:
            break
        headers.remove(best[0])

        G.add_node(best[0])
        if GG is not None:
            G.add_edge(GG, best[0], label="XfghfgdfghfghD")

        st = set(data[best[0]])
        for i in st:
            sub_data = data.loc[data[best[0]] == i]
            # if sub_data.shape[0] < 2:
            #     continue
            if GG is None:
                treeGen(sub_data, headers.copy(), G, best[0])
            else:
                treeGen(sub_data, headers.copy(), G, str(best[0]) + str(i))

    # G = nx.Graph()
    # G.add_node(4)
    # G.add_edge(1, 2)  # default edge data=1
    # G.add_edge(2, 3, weight=0.1)  # specify edge data
    #
    # nx.draw(G, with_labels=True, font_weight='bold')
    #
    # plt.show()


if __name__ == '__main__':
    file = "titanic-homework.csv"
    datat = pd.read_csv(file)
    columnn = 'Pclass'

    # print(entropy(data, column))
    # conditionalEntropy(data, column)
    # print(gain(data, column))
    G = nx.Graph()
    GG = None
    treeGen(datat, list(datat.columns.values), G, GG)
    nx.draw(G, with_labels=True, font_weight='bold')

    plt.show()
