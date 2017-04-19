#!/usr/bin/python3

import sys
import csv
import matplotlib.pyplot as plt

'''
conversion of discrete non-numerical data by label:

'''

#global variables
path = '../../data/middle east/middle_east.csv'
output = 'graph.png'
translate = {'F': 0, 'M': 1, 'S': 1, 'KW': 0, 'lebanon': 1, 'KW': 1,
'Egypt': 2, 'Saudi Arabia': 3, 'USA': 4, 'Jordan': 5, 'venezuala': 6,
'Iran': 7, 'Tunis': 8, 'Syria': 9, 'Morocco': 10, 'Palestine': 11,
'Iraq':12, 'Lybia': 13, 'KuwaIT': 0, 'SaudiArabia': 3, 'lowerlevel': 0,
'MiddleSchool': 1, 'HighSchool': 2, 'G-01': 1, 'G-02': 2, 'G-03': 3,
'G-04': 4, 'G-05': 5, 'G-06': 6, 'G-07': 7, 'G-08': 8, 'G-09': 9,
'G-10': 10, 'G-11': 11, 'G-12': 12, 'A': 0, 'B': 1, 'C': 2, 'IT': 0,
'Math':1, 'Arabic': 2, 'Science': 3, 'English': 4, 'Quaran': 5,
'Spanish': 6, 'French': 7, 'History': 8, 'Biology': 9, 'Chemistry': 10,
'Geology': 11, 'Mum': 0, 'Father': 1, 'No': 0, 'Yes': 1, 'Bad': 0,
'Good': 1, 'Under-7': 0, 'Above-7': 1, 'L': 0, 'H': 2}


def main(args):

    if len(args) != 3:
        print("run as: ./mideast-visual.py attribute1 attribute2")

    #read in data and seperate labels
    infile = open(path, 'r')
    dataReader = csv.reader(infile, delimiter=',')
    raw = []
    for row in dataReader:
        raw.append(row)

    labels = raw[0]
    data = raw[1:]

    #convert discrete data
    r = 0
    for row in data:
        c = 0
        for item in row:
            if item.isdigit():
                data[r][c] = int(item)
            else:
                data[r][c] = translate.get(item, -1)
            c += 1
        r += 1

    #seperate the data to plot
    x = -1
    y = -1
    for i in range(len(labels)):
        if args[1] == labels[i]:
            x = i
        elif args[2] == labels[i]:
            y = i
    item1 = []
    item2 = []
    i = 0
    for row in data:
        item1.append(row[x])
        item2.append(row[y])
        i += 1

    #make plot
    plt.scatter(item1, item2, label='Visual')
    plt.xlabel(labels[x])
    plt.ylabel(labels[y])
    plt.savefig(output)

if __name__ == '__main__':
    main(sys.argv)
