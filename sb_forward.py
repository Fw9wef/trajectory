import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from env import Drones
from tqdm import tqdm
from stable_baselines3 import PPO
from env_options import MAX_STEPS, AGENT_INDEX
#from sb_train import FE
import cv2
matplotlib.use("Agg")


path_to_data = "./" + str(AGENT_INDEX) + "/"
path_to_weights = "./sb_ppo_drones"
path_to_video_frames = "./frames/"
max_steps = MAX_STEPS
test_env = Drones(debug=True)
obs = test_env.reset()

#policy_kwargs = dict(features_extractor_class=FE)
#controller = PPO("MlpPolicy", test_env, policy_kwargs=policy_kwargs)
controller = PPO.load(path_to_weights, env=test_env)

def get_action(action=None, observation=None):
    if action:
        return action
    else:
        action = controller.predict(observation, deterministic=True)[0]
        return action


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


def save_img(i, plan, plan_detected, plan_camera, d1, d1_detected, d1_camera, d2, d2_detected, d2_camera):
    plan_y, plan_x = plan
    d1y, d1x = d1
    d2y, d2x = d2
    plt.figure(figsize=(25, 25))
    plt.ylim(-700000, 100000)
    plt.xlim(-100000, 700000)
    plt.plot(plan_y * 100, plan_x * 100, c='g')
    plt.plot(d1y * 100, d1x * 100, c='y')
    plt.plot(d2y * 100, d2x * 100, c='k')
    plt.scatter(objs[:, 1], objs[:, 0], c='b', marker='x')
    if len(plan_detected):
        plt.scatter(plan_detected[:, 1] * 100, plan_detected[:, 0] * 100, c='g', marker='+')
    if len(d1_detected):
        plt.scatter(d1_detected[:, 1] * 100, d1_detected[:, 0] * 100, c='y', marker='1')
    if len(d2_detected):
        plt.scatter(d2_detected[:, 1] * 100, d2_detected[:, 0] * 100, c='k', marker='2')

    plt.plot(plan_camera[:, 0] * 100, plan_camera[:, 1] * 100, c='g')
    plt.plot(d1_camera[:, 0] * 100, d1_camera[:, 1] * 100, c='y')
    plt.plot(d2_camera[:, 0] * 100, d2_camera[:, 1] * 100, c='k')
    plt.savefig(os.path.join(path_to_video_frames, str(i)+".png"))
    plt.close("all")


plan_uav_pos = list()
d1_uav_pos = list()
d2_uav_pos = list()
plan_detected = set()
d1_detected = set()
d2_detected = set()

objs, start, waypoints = load()

img_n = 0
filenames = list()

message = dict()
message["need_new_action"] = True
for step_num in tqdm(range(1, max_steps+1)):
    if message["need_new_action"]:
        action = get_action(action=None, observation=obs)
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

    if step_num % 10 == 0:
        curr_plan = np.asarray(plan_uav_pos)
        curr_plan = curr_plan[:, 0], curr_plan[:, 1]
        curr_d1 = np.asarray(d1_uav_pos)
        curr_d1 = curr_d1[:, 0], curr_d1[:, 1]
        curr_d2 = np.asarray(d2_uav_pos)
        curr_d2 = curr_d2[:, 0], curr_d2[:, 1]
        curr_plan_detected = np.asarray(list(plan_detected))
        curr_d1_detected = np.asarray(list(d1_detected))
        curr_d2_detected = np.asarray(list(d2_detected))

        cameras = message["cams"]

        save_img(img_n, curr_plan, curr_plan_detected, cameras["plan"],
                 curr_d1, curr_d1_detected, cameras["d1"],
                 curr_d2, curr_d2_detected, cameras["d2"])
        filenames.append("gif/"+str(img_n)+".png")
        img_n += 1


plan_uav_pos = np.asarray(plan_uav_pos)
d1_uav_pos = np.asarray(d1_uav_pos)
d2_uav_pos = np.asarray(d2_uav_pos)

plan_target_y = waypoints[:, 1]
plan_target_x = waypoints[:, 0]
plan_start_pos_y = start[0, 1]
plan_start_pos_x = start[0, 0]
plan_x = plan_uav_pos[:, 1]
plan_y = plan_uav_pos[:, 0]
d1x = d1_uav_pos[:, 1]
d1y = d1_uav_pos[:, 0]
d2x = d2_uav_pos[:, 1]
d2y = d2_uav_pos[:, 0]

plan_detected = np.asarray(list(plan_detected))
d1_detected = np.asarray(list(d1_detected))
d2_detected = np.asarray(list(d2_detected))

plt.figure(figsize=(25, 25))
plt.plot(plan_y*100, plan_x*100, c='g')
plt.plot(d1y*100, d1x*100, c='y')
plt.plot(d2y*100, d2x*100, c='k')
plt.scatter(objs[:, 1], objs[:, 0], c='b', marker='x')
plt.scatter([plan_start_pos_y], [plan_start_pos_x], c='r')
if len(plan_detected):
    plt.scatter(plan_detected[:, 1]*100, plan_detected[:, 0]*100, c='g', marker='+')
if len(d1_detected):
    plt.scatter(d1_detected[:, 1]*100, d1_detected[:, 0]*100, c='y', marker='1')
if len(d2_detected):
    plt.scatter(d2_detected[:, 1]*100, d2_detected[:, 0]*100, c='k', marker='2')
plt.savefig("plan_navigation.png")

video_name = 'nav.avi'

frame = cv2.imread(filenames[0])
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps=5, frameSize=(width, height))

for filename in filenames[1:]:
    video.write(cv2.imread(filename))

cv2.destroyAllWindows()
video.release()
