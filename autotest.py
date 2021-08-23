import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from env import Drones
from tqdm import tqdm
from itertools import chain


path_to_data = "./0/"
max_steps = 1000
test_env = Drones(debug=True)
test_env.reset()

def get_action(action=None):
    if action:
        return action
    else:
        return [1 for _ in range(4)]


def load():
    path_to_objs = os.path.join(path_to_data, 'objects.csv')
    objs = pd.read_csv(path_to_objs, sep=";")
    objs = objs.values[:, :2]

    path_to_start = os.path.join(path_to_data, 'start.csv')
    start = pd.read_csv(path_to_start, sep=";")
    start = start.values[:, :2]

    path_to_plan_waypoints = os.path.join(path_to_data, 'waypoints.csv')
    waypoints = pd.read_csv(path_to_plan_waypoints, sep=";")
    waypoints = waypoints.values[:, :2]
    return objs, start, waypoints

plan_uav_pos = list()
d1_uav_pos = list()
d2_uav_pos = list()
plan_detected = set()
d1_detected = set()
d2_detected = set()

objs, start, waypoints = load()

action = [objs[0, 0], objs[0, 1], objs[1, 0], objs[1, 1]]

for step_num in tqdm(range(1, max_steps+1)):
    action = get_action(action)
    obs, step_reward, done, message = test_env.step(action)

    uav_coords = obs["uav"]
    plan_uav_pos.append(uav_coords[0, :2])
    d1_uav_pos.append(uav_coords[1, :2])
    d2_uav_pos.append(uav_coords[2, :2])

    if "plan" in message.keys():
        plan_detected.add(message["plan"])
    if 2 in message.keys():
        d1_detected.add(message[2])
    if 3 in message.keys():
        d2_detected.add(message[3])

    if done:
        break

plan_uav_pos = np.asarray(plan_uav_pos)
d1_uav_pos = np.asarray(d1_uav_pos)
d2_uav_pos = np.asarray(d2_uav_pos)

plan_target_x = waypoints[:, 1]
plan_target_y = waypoints[:, 0]
plan_start_pos_x = start[0, 1]
plan_start_pos_y = start[0, 0]
plan_x = plan_uav_pos[:, 1]
plan_y = plan_uav_pos[:, 0]

xx = [x for x in chain([plan_start_pos_x], plan_target_x)]
yy = [x for x in chain([plan_start_pos_y], plan_target_y)]

plan_detected = np.asarray(list(plan_detected))

plt.figure(figsize=(25, 25))
plt.plot(xx, yy, c='b')
plt.plot(plan_x, plan_y, c='r')
plt.scatter(objs[:, 1], objs[:, 0], c='b')
plt.scatter([plan_start_pos_x], [plan_start_pos_y], c='r')
plt.scatter(plan_detected[:, 1], plan_detected[:, 0], c='g')
plt.savefig("plan_navigation.png")
