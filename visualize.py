import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def read_drone_trajectory(path):
    y_coords = list()
    x_coords = list()
    file = open(path)
    for line in file:
        y, x = [float(a) for a in line.strip().split(';')]
        y_coords.append(y)
        x_coords.append(x)
    return x_coords, y_coords


def read_objs(path):
    y_coords = list()
    x_coords = list()
    file = open(path)
    for line in file:
        y, x = [float(a) for a in line.strip().split("   ")]
        y_coords.append(y/100)
        x_coords.append(x/100)
    return x_coords, y_coords


glob_path = "./"
drone_files = ["drone"+str(i)+".txt" for i in range(1, 5)]
detect_files = ["drone"+str(i)+"_detect.txt" for i in [3,4]]
drone_paths = [os.path.join(glob_path, file) for file in drone_files]
detect_paths = [os.path.join(glob_path, file) for file in detect_files]
path_to_points = os.path.join(glob_path, "new_detect_list.txt")


plt.figure(figsize=(16, 16))
for path, color in zip(drone_paths, ['r', 'g', 'b', 'y']):
    xx, yy = read_drone_trajectory(path)
    plt.scatter(xx, yy, c=color, marker=".", alpha=0.01)

for path, color in zip(detect_paths, ['b', 'y']):
    xx, yy = read_drone_trajectory(path)
    plt.scatter(xx, yy, c=color, marker="*", linewidths=3)

xx, yy = read_objs(path_to_points)
plt.scatter(yy, xx, marker='P')

plt.savefig("course.png")
