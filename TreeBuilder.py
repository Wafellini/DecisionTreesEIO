from math import log2

from treelib import Tree
import pandas as pd


class TreeBuilder:
    __decision_class_name: str
    __node_repeats_count = {}

    def __init__(self, decision_class: str):
        self.__decision_class_name = decision_class

    def __entropy(self, data: pd.DataFrame) -> float:
        lst = list(data[self.__decision_class_name])
        st = set(lst)

        entrpy = 0
        for i in st:
            entrpy -= log2(lst.count(i) / len(lst)) * lst.count(i) / len(lst)

        return entrpy

    def __conditional_entropy(self, data: pd.DataFrame, column: str) -> float:
        st = set(data[column])

        con_entr = 0
        for i in st:
            subset = data.loc[data[column] == i]
            con_entr += self.__entropy(subset) * subset.shape[0] / data.shape[0]

        return con_entr

    def __gain(self, data: pd.DataFrame, column: str) -> float:
        return self.__entropy(data) - self.__conditional_entropy(data, column)

    def __gain_ratio(self, data: pd.DataFrame, column: str) -> float:
        st = set(data[column])

        intr_info = 0
        for i in st:
            lst = data.loc[data[column] == i]
            intr_info -= log2(lst.shape[0] / data.shape[0]) * lst.shape[0] / data.shape[0]

        if intr_info == 0:
            return 0.00001
        return self.__gain(data, column) / intr_info

    def __choose_best_attribute(self, data: pd.DataFrame, headers: list) -> list:
        best = [0, 0]
        for attribute in headers[0:-1]:
            tmp = self.__gain_ratio(data, attribute)
            if tmp > best[1]:
                best = [attribute, tmp]

        return best

    def __increment_counter(self, name: str):
        if name in self.__node_repeats_count:
            self.__node_repeats_count[name] += 1
        else:
            self.__node_repeats_count[name] = 0

    def __get_node_id(self, name: str) -> str:
        self.__increment_counter(name)
        return name + "_" + str(self.__node_repeats_count[name])

    def tree_gen(self, data: pd.DataFrame, headers: list) -> Tree:
        tree = Tree()

        decision_class_values = list(data[self.__decision_class_name])
        if len(set(decision_class_values)) == 1:
            value = str(decision_class_values[0])
            tree.create_node(value, self.__get_node_id(value))
            return tree

        best_attribute = self.__choose_best_attribute(data, headers)
        if best_attribute[0] != 0:
            headers.remove(best_attribute[0])
        else:
            best = [0, 0]
            for i in set(decision_class_values):
                element_count = decision_class_values.count(i)
                if element_count > best[1]:
                    best[0] = i

            value = str(best[0])
            tree.create_node(value, self.__get_node_id(value))
            return tree

        root_id = self.__get_node_id(best_attribute[0])
        tree.create_node(best_attribute[0], root_id)
        possible_values = set(data[best_attribute[0]])
        for i in possible_values:
            sub_data = data[data[best_attribute[0]] == i]
            sub_tree = self.tree_gen(sub_data, headers)

            tree.paste(root_id, sub_tree)

        return tree
