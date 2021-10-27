import random
from math import log

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

#from stack
def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

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
        node = "Prev choice: " + prev_choice + "\nVal: " + str(best[0]) + "\n" + str(random.randint(1, 1000000))
        G.add_edge(prev_node, node, weight=1, label=prev_choice)

    st = set(data[best[0]])
    for i in st:
        sub_data = data.loc[data[best[0]] == i]
        if len(set(sub_data["Survived"])) == 1:
            G.add_edge(node, "Survived: " +
                   str(list(sub_data["Survived"])[0]) + "\nPrev choice: " + str(i) + " " + str(random.randint(1, 1000000)), weight=1,
                   label=prev_choice)
            continue
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

    G = nx.DiGraph()
    prev_node = None
    datat, headers = normalizeData(datat)

    treeGen(datat, headers, G, prev_node, None)
    pos = hierarchy_pos(G, "Sex")

    nx.draw(G, pos=pos, with_labels=True)

    plt.show()
