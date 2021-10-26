from math import log

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from TreeBuilder import TreeBuilder


def entropy(data: pd.DataFrame, column: str) -> float:
    lst = list(data[column])
    st = set(lst)

    entrpy = 0
    for i in st:
        entrpy -= log(lst.count(i) / len(lst), 2) * lst.count(i) / len(lst)

    return entrpy


def conditional_entropy(data: pd.DataFrame, column: str) -> float:
    st = set(data[column])

    con_entr = 0
    for i in st:
        subset = data.loc[data[column] == i]
        con_entr += entropy(subset, "Survived") * subset.shape[0] / data.shape[0]

    return con_entr


def gain(data: pd.DataFrame, column: str) -> float:
    return entropy(data, "Survived") - conditional_entropy(data, column)


def gain_ratio(data: pd.DataFrame, column: str) -> float:
    st = set(data[column])

    intr_info = 0
    for i in st:
        lst = data.loc[data[column] == i]
        intr_info -= log(lst.shape[0] / data.shape[0], 2) * lst.shape[0] / data.shape[0]

    if intr_info == 0:
        return 0.00001
    return gain(data, column) / intr_info


def choose_best_attribute(data: pd.DataFrame, headers: list, decision_class) -> list:
    best = [0, 0]
    for attribute in headers[0:-1]:
        # if data.shape[0] == 1:
        #     continue
        tmp = gain_ratio(data, attribute)
        if tmp > best[1]:
            best = [attribute, tmp]

    return best


def tree_gen(data: pd.DataFrame, headers: list, decision_class: str) -> nx.Graph:
    tree = nx.Graph()

    decision_class_values = data[decision_class]
    if len(set(decision_class_values)) == 1:
        tree.add_node(decision_class_values[0])
        return tree

    best_attribute = choose_best_attribute(data, headers)

    while len(headers) > 1:
        if len(set(data["Survived"])) == 1:
            break

        best_attribute = choose_best_attribute(data, headers)

        if best_attribute[0] == 0:
            break
        headers.remove(best_attribute[0])

        G.add_node(best_attribute[0])
        # if GG is not None:
        #     G.add_edge(GG, best_attribute[0], label="XfghfgdfghfghD")

        possible_values = set(data[best_attribute[0]])
        for i in possible_values:
            sub_data = data.loc[data[best_attribute[0]] == i]
            if GG is None:
                tree_gen(sub_data, headers.copy(), G, best[0])
            else:
                tree_gen(sub_data, headers.copy(), G, str(best[0]) + str(i))


if __name__ == '__main__':
    file = "titanic-homework.csv"
    datat = pd.read_csv(file)

    tree_builder = TreeBuilder("Survived")

    G = tree_builder.tree_gen(datat, list(datat.columns.values))
    G.show()
    # GG = None
    # tree_gen(datat, list(datat.columns.values), G, GG)
    # nx.draw(G, with_labels=True, font_weight='bold')

    #plt.show()
