"""Decode the word encoded by the splitting features of a fitted decision tree.

Usage:
    python main_snowpit.py                 # test-input.csv, otherwise train-input.csv
    python main_snowpit.py my-data.csv
"""
import csv
import os
import sys

from sklearn.tree import DecisionTreeClassifier


def decode(csv_path):
    with open(csv_path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if not columns or len(columns) < 2:
            raise ValueError('Expected feature columns and a target column')
        target = 'slab_released' if 'slab_released' in columns else columns[-1]
        features = [column for column in columns if column != target]
        X, y = [], []
        for row in reader:
            X.append([float(row[name]) for name in features])
            y.append(int(row[target]))

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)

    # sklearn stores nodes in depth-first, left-before-right (preorder).
    # Explicitly traverse, so the reading order is not an undocumented assumption.
    tree = clf.tree_
    letters = []
    stack = [0]
    while stack:
        node = stack.pop()
        if tree.children_left[node] == tree.children_right[node]:
            continue  # leaf
        feature_name = features[tree.feature[node]]
        letters.append(feature_name[0].upper())
        stack.append(int(tree.children_right[node]))
        stack.append(int(tree.children_left[node]))

    return 'SIGMOID_' + ''.join(letters)


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    elif os.path.exists('test-input.csv'):
        filename = 'test-input.csv'
    else:
        filename = 'train-input.csv'
    print(decode(filename))


if __name__ == '__main__':
    main()
