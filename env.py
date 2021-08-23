import ctypes
from ctypes import *
import gym
from gym import spaces
import numpy as np
import pandas as pd
import os

EVENTS = {
    0: "nothing",
    1: "new_obs_point",
    2: "waypoint_reached",
    3: "1+2",
    4: "obs_point_searched",
    5: "mc"
}


class StepEvents(ctypes.Structure):
    _fields_ = [('val1', ctypes.c_int),
                ('val2', ctypes.c_int),
                ('val3', ctypes.c_int),
                ('val4', ctypes.c_int)]


class Pos2D(ctypes.Structure):
    _fields_ = [('x', ctypes.c_double),
                ('y', ctypes.c_double)]


class DroneState(ctypes.Structure):
    _fields_ = [('pos', Pos2D),
                ('h', ctypes.c_double),
                ('course', ctypes.c_double),
                ('norm_course', ctypes.c_double)]


course_arr = Pos2D * 4


class RandomPointsGenerator:
    def __init__(self, left_bottom, right_top, plan_uav_vision_width,
                 plan_uav_vision_high, plan_uav_overlap, path_to_folder, x100=True):
        if x100:
            left_bottom = [x*100 for x in left_bottom]
            right_top = [x * 100 for x in right_top]
        self.left_bottom = left_bottom
        self.right_top = right_top
        self.y_min, self.x_min = left_bottom
        self.y_max, self.x_max = right_top
        self.y_delta, self.x_delta = abs(self.y_max - self.y_min), abs(self.x_max - self.x_min)
        self.delta = np.asarray([self.y_delta, self.x_delta])
        self.bias = np.asarray([self.y_min, self.x_min])
        self.plan_uav_vision_width = plan_uav_vision_width
        self.plan_uav_vision_hight = plan_uav_vision_high
        self.plan_uav_overlap = plan_uav_overlap
        self.path_to_folder = path_to_folder
        if not os.path.isdir(path_to_folder):
            os.mkdir(path_to_folder)

        self.objects = list()
        self.plan_uav_waypoints = list()
        self.rl_uav_waypoints = list()
        self.start_points = dict()

    def __call__(self, n_range):
        self.get_n_random_points(n_range)
        self.get_start_positions()
        self.get_plan_uav_waypoints()
        self.get_rl_uav_waypoint()
        self.normalize_coordinates()
        self.save_start_data()
        return {
            'objs': self.objects,
            'plan_waypoints': self.plan_uav_waypoints,
            'start_points': self.start_points
        }

    @staticmethod
    def baseline_place_points(n_regions, per_region_points, per_std, w=1, h=1):
        objects = []
        for _ in range(n_regions):
            n_points = np.round(np.random.uniform(per_region_points[0], per_region_points[1], 1)).astype(np.int32)[0]
            std = np.random.uniform(per_std[0], per_std[1], n_points)[0]
            cx, cy = np.random.uniform(0, 1, 2)
            cx, cy = cx * w, cy * h

            xx = np.random.normal(scale=std, size=n_points) + cx
            yy = np.random.normal(scale=std, size=n_points) + cy

            for x, y in zip(xx, yy):
                objects.append([y, x])

        return np.array(objects)

    @staticmethod
    def random_over_random():
        n = np.random.randint(1, 5)
        np_min = 5 + np.random.randint(0, 20)
        np_max = np_min + np.random.randint(0, 25 - np_min)
        s_min = 0.05 + np.random.rand() * 0.15
        s_max = s_min + np.random.rand() * (0.2 - s_min)
        return RandomPointsGenerator.baseline_place_points(n, [np_min, np_max], [s_min, s_max], w=1, h=1)

    def get_n_random_points(self, n_range):
        n = np.random.randint(n_range[0], n_range[1])
        objects = RandomPointsGenerator.random_over_random()
        objects = np.random.permutation(objects)[:n]
        self.objects = np.asarray(objects)

    @staticmethod
    def waypoint_up(curr_y, curr_x, vertical_run_l):
        return curr_y + vertical_run_l, curr_x

    @staticmethod
    def waypoint_down(curr_y, curr_x, vertical_run_l):
        return curr_y - vertical_run_l, curr_x

    @staticmethod
    def waypoint_right(curr_y, curr_x, horizontal_run_l):
        return curr_y, curr_x + horizontal_run_l

    @staticmethod
    def waypoint_left(curr_y, curr_x, horizontal_run_l):
        return curr_y, curr_x - horizontal_run_l

    def get_plan_uav_waypoints(self):
        left, right, bot, top = np.inf, -np.inf, np.inf, -np.inf
        for y, x in self.objects:
            left = x if x < left else left
            right = x if x > left else right
            bot = y if y < bot else bot
            top = y if y > top else top

        vertical_run_l = top - bot
        horizontal_run_l = self.plan_uav_vision_width * self.plan_uav_overlap

        start_y, start_x = self.start_points['plan']
        movement_list = list()

        if abs(start_y - bot) < abs(start_y - top):
            curr_waypoint_y = bot
            movement_list.append('go_top')
        else:
            curr_waypoint_y = top
            movement_list.append('go_bot')

        if abs(start_x - left) < abs(start_x - right):
            curr_waypoint_x = left
            movement_list.append('go_right')
            stop_condition = 'right'
        else:
            curr_waypoint_x = right
            movement_list.append('go_left')
            stop_condition = 'left'

        if movement_list[0] == 'go_top':
            movement_list.append('go_bot')
        else:
            movement_list.append('go_top')

        if movement_list[1] == 'go_right':
            movement_list.append('go_left')
        else:
            movement_list.append('go_right')

        waypoints = [(curr_waypoint_y, curr_waypoint_x)]
        if stop_condition == 'right':
            condition = lambda inp: inp < right
        elif stop_condition == 'left':
            condition = lambda inp: inp > left

        n_step = 0
        while condition(curr_waypoint_x):
            movement = movement_list[n_step % 4]
            if movement == 'go_top':
                curr_waypoint_y, curr_waypoint_x = self.waypoint_up(curr_waypoint_y, curr_waypoint_x,
                                                                    vertical_run_l)
            elif movement == 'go_bot':
                curr_waypoint_y, curr_waypoint_x = self.waypoint_down(curr_waypoint_y, curr_waypoint_x,
                                                                      vertical_run_l)
            elif movement == 'go_right':
                curr_waypoint_y, curr_waypoint_x = self.waypoint_right(curr_waypoint_y, curr_waypoint_x,
                                                                       horizontal_run_l)
            elif movement == 'go_left':
                curr_waypoint_y, curr_waypoint_x = self.waypoint_left(curr_waypoint_y, curr_waypoint_x,
                                                                      horizontal_run_l)
            waypoints.append((curr_waypoint_y, curr_waypoint_x))

        exit_x = 0 if curr_waypoint_x < 0.5 else 1
        exit_y = 0 if curr_waypoint_y < 0.5 else 1
        waypoints.append((exit_y, exit_x))
        self.plan_uav_waypoints = waypoints

    def get_rl_uav_waypoint(self):
        start_y, start_x = self.start_points['rl']
        corner_x = 0.1 if start_x < 0.5 else 0.9
        corner_y = 0.1 if start_y < 0.5 else 0.9
        exit_corner_x = 0.1 if corner_x > 0.5 else 0.9
        waypoints = [(corner_y, corner_x), (corner_y, exit_corner_x)]
        self.rl_uav_waypoints = waypoints

    def get_start_positions(self):
        for key in ['plan', 'rl', 'd1', 'd2']:
            point = np.random.rand()
            if point < 0.25:
                start_x, start_y = point*4, 0.01
            elif point < 0.5:
                start_x, start_y = 0.99, (point - 0.25) * 4
            elif point < 0.75:
                start_x, start_y = (point - 0.5) * 4, 0.99
            else:
                start_x, start_y = 0.01, (point - 0.75) * 4

            self.start_points[key] = (start_y, start_x)

    def normalize_coordinates(self):
        self.objects = np.asarray(self.objects) * self.delta + self.bias
        self.plan_uav_waypoints = np.asarray(self.plan_uav_waypoints) * self.delta + self.bias
        self.rl_uav_waypoints = np.asarray(self.rl_uav_waypoints) * self.delta + self.bias
        for key, value in self.start_points.items():
            self.start_points[key] = value * self.delta + self.bias

    def save_start_data(self):
        self.save_objects()
        self.save_plan_waypoints()
        self.save_rl_waypoints()
        self.save_start_positions()

    def save_objects(self):
        data = {
            'Latitude': self.objects[:,0],
            'Longitude': self.objects[:,1],
            'Altitude': [200 for _ in range(self.objects.shape[0])],
            'class': ['car' for _ in range(self.objects.shape[0])]
        }
        data = pd.DataFrame(data)
        path_to_csv = os.path.join(self.path_to_folder, "objects.csv")
        data.to_csv(path_to_csv, index=False, sep=';')

    def save_plan_waypoints(self):
        data = {
            'Latitude': self.plan_uav_waypoints[:, 0],
            'Longitude': self.plan_uav_waypoints[:, 1],
            'Altitude': [200 for _ in range(self.plan_uav_waypoints.shape[0])],
        }
        data = pd.DataFrame(data)
        path_to_csv = os.path.join(self.path_to_folder, "waypoints.csv")
        data.to_csv(path_to_csv, index=False, sep=';')

    def save_rl_waypoints(self):
        data = {
            'Latitude': self.rl_uav_waypoints[:, 0],
            'Longitude': self.rl_uav_waypoints[:, 1],
            'Altitude': [200 for _ in range(self.rl_uav_waypoints.shape[0])],
        }
        data = pd.DataFrame(data)
        path_to_csv = os.path.join(self.path_to_folder, "waypoints_RL.csv")
        data.to_csv(path_to_csv, index=False, sep=';')

    def save_start_positions(self):
        start_points = list()
        for key in ['plan', 'rl', 'd1', 'd2']:
            start_points.append(self.start_points[key])
        data = {
            'Latitude': start_points[:, 0],
            'Longitude': start_points[:, 1],
            'Altitude': [200 for _ in range(start_points.shape[0])],
        }
        data = pd.DataFrame(data)
        path_to_csv = os.path.join(self.path_to_folder, "start.csv")
        data.to_csv(path_to_csv, index=False, sep=';')


class Drones(gym.Env):
    def __init__(self, work_dir="./", max_ooi=100, max_steps=5000, n_range=[80,100], agent_index=0,
                 time_penalty=1e-2, survey_reward=2, mc_reward=10, left_bottom=[0, 0], right_top=[6000, 6000],
                 plan_uav_vision_width=0.1, plan_uav_vision_high=0.1, plan_uav_overlap=0.1, debug=False):
        super(Drones, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.max_ooi = max_ooi
        self.max_steps = max_steps
        self.n_range = n_range
        self.time_penalty = time_penalty
        self.survey_reward = survey_reward
        self.mc_reward = mc_reward
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict({'ooi_coords': spaces.Box(-1, 1, (max_ooi, 2), np.float32),
                                              'ooi_mask': spaces.MultiBinary((max_ooi,)),
                                              'uav': spaces.Box(-1, 1, (3, 4), np.float32)})

        self.work_dir = work_dir
        self.agent_index = str(agent_index)
        self.agent_dir = os.path.join(self.work_dir, self.agent_index)
        self.agent_dir = bytes(self.agent_dir, encoding="utf-8")
        if not os.path.isdir(self.agent_dir):
            os.mkdir(self.agent_dir)
        agent_dir = list(self.agent_dir)
        chars = [c_char(x) for x in agent_dir]
        self.char_m = c_char * len(agent_dir)
        self.agent_dir_c = self.char_m(*chars)
        self.left_bottom, self.right_top = np.asarray(left_bottom), np.asarray(right_top)
        self.high, self.width = self.right_top - self.left_bottom
        self.exp_init = RandomPointsGenerator(left_bottom, right_top, plan_uav_vision_width,
                                              plan_uav_vision_high, plan_uav_overlap,
                                              path_to_folder=self.agent_dir)

        self.testpp = cdll.LoadLibrary("../dll/Aurora__1_model_solution.dll")
        self.test = "uninitialized"
        self.load_library()

        self.oois = list()
        self.min_dist = np.inf
        self.plan_waypoints = list()
        self.active_oois = set()
        self.total_steps = 0
        self.surveyed_oois = 0
        self.plan_n_detected = 0
        self.d1_n_detected = 0
        self.d2_n_detected = 0

        self.debug = debug

    def step(self, action):
        step_reward = 0.
        self.set_course(action)

        break_flag = False
        while True:
            self.total_steps += 1
            step_reward -= self.time_penalty

            self.update_cameras()
            events = self.simulation_step()

            # new active ooi
            if events[0] == 1:
                self.plan_n_detected += 1
                new_ooi = self.get_latest_plan_ooi()
                self.active_oois.add(new_ooi)
                break_flag = True

            # waypoint reached
            if 2 in events[2:]:
                break_flag = True

            # ooi surveyed
            for i, e in enumerate(events[2:], start=2):
                if e == 4:

                    if i == 2:
                        self.d1_n_detected += 1
                    elif i == 3:
                        self.d2_n_detected += 1

                    last_surveyed_ooi = self.get_last_surveyed_ooi(uav_index=2)
                    unique, gt_ooi = self.if_active_ooi(last_surveyed_ooi)
                    if unique:
                        self.remove_active_ooi(gt_ooi)
                        break_flag = True
                        step_reward += self.survey_reward
                        self.surveyed_oois += 1

            if break_flag:
                break

        # check for mission complete
        if self.surveyed_oois == len(self.oois):
            done = True
            step_reward += self.mc_reward
        else:
            done = False

        obs = self._next_observation()

        return obs, step_reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        exp_data = self.exp_init(self.n_range, self.debug)
        self.testpp.init_c(self.test, self.agent_dir_c)
        self.oois = exp_data['objs']
        self.min_dist = np.inf
        self.plan_waypoints = exp_data['plan_waypoints']
        self.active_oois = set()
        self.total_steps = 0
        self.surveyed_oois = 0
        self.plan_n_detected = 0
        self.d1_n_detected = 0
        self.d2_n_detected = 0

    def render(self, mode='human', close=False):
        pass

    def close(self):
        self.testpp.destruct_c(self.test)

    def load_library(self):
        # define C funcs arg types and return types
        self.testpp.init_c.argtypes = [ctypes.c_void_p, POINTER(self.char_m)]
        self.testpp.step_c.argtypes = [ctypes.c_void_p]
        self.testpp.reset_c.argtypes = [ctypes.c_void_p]
        self.testpp.getState_c.argtypes = [ctypes.c_void_p]
        self.testpp.getObjectState_c.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.testpp.getObjectState_SIR_STV_с.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.testpp.getVisionState_c.argtypes = [ctypes.c_void_p]
        self.testpp.setCourse_c.argtypes = [ctypes.c_void_p, POINTER(course_arr)]
        self.testpp.setVisionCourse_c.argtypes = [ctypes.c_void_p, POINTER(course_arr)]
        self.testpp.saveTrackToFile_c.argtypes = [ctypes.c_void_p]
        self.testpp.destructor_c.argtypes = [ctypes.c_void_p]

        self.testpp.export_class_c.restype = ctypes.c_void_p
        self.testpp.step_c.restype = ctypes.c_void_p
        self.testpp.reset_c.restype = ctypes.c_void_p
        self.testpp.getState_c.restype = ctypes.c_void_p
        self.testpp.getVisionState_c.restype = ctypes.c_void_p
        self.testpp.getObjectState_c.restype = ctypes.c_void_p
        self.testpp.getObjectState_SIR_STV_с.restype = ctypes.c_void_p

        self.test = self.testpp.export_class_c()

    def find_min_dist_btw_oois(self):
        min_dist = np.inf
        for i in range(len(self.oois)):
            for j in range(i+1, len(self.oois)):
                x, y = self.oois[i], self.oois[j]
                dist = np.sqrt(np.sum((x - y) ** 2))
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def simulation_step(self):
        step_events_c = self.testpp.step_c(self.test)
        step_events = StepEvents.from_address(step_events_c)
        events = [step_events.val1, step_events.val2, step_events.val3, step_events.val4]
        return events

    def norm_coords(self, y, x):
        y = self.left_bottom[0] + self.high * (y+1) / 2
        x = self.left_bottom[1] + self.width * (x + 1) / 2
        return y, x

    def set_course(self, coords):
        uav1_y, uav1_x, uav2_y, uav2_x = coords
        (uav1_y, uav1_x), (uav2_y, uav2_x) = self.norm_coords(uav1_y, uav1_x), self.norm_coords(uav2_y, uav2_x)
        course1, course2 = Pos2D(uav1_x, uav1_y), Pos2D(uav2_x, uav2_y)
        course = [course1, course2, course1, course2]
        course_c = course_arr(*course)
        self.testpp.setCourse_c(self.test, course_c)

    def update_cameras(self):
        _, _, uav1, uav2 = self.get_uav_coords()
        objects = self.get_ooi()
        vision_objs = list()
        for uav in enumerate([uav1, uav2]):
            obj_coords = self.find_nearest_ooi(uav, objects)
            vision_objs.append(obj_coords)
        self.set_vision_course(vision_objs)

    def find_nearest_ooi(self, uav_coords, objs_list):
        if len(objs_list):
            uav = np.asarray(uav_coords)
            objs = np.asarray(objs_list)
            delta = np.sqrt(np.sum((objs - uav) ** 2, axis=-1))
            nearest_obj_ind = np.argmin(delta)
            nearest_obj = objs_list[nearest_obj_ind]
        else:
            nearest_obj = (1, 1)
        return nearest_obj

    def set_vision_course(self, vision_objs):
        coords1, coords2 = vision_objs
        coords1, coords2 = self.norm_coords(*coords1), self.norm_coords(*coords2)
        coords1, coords2 = Pos2D(coords1[1], coords1[0]), Pos2D(coords2[1], coords2[0])
        coords = [coords1, coords2, coords1, coords2]
        coords_c = course_arr(*coords)
        self.testpp.setVisionCourse_c(self.test, coords_c)

    def _next_observation(self):
        ooi_coords, ooi_mask = self.get_ooi_and_mask()
        uav_coords = self.get_uav_coords()
        obs = {'ooi_coords': np.asarray(ooi_coords, dtype=np.flaot32),
               'ooi_mask': np.asarray(ooi_mask, dtype=np.int8),
               'uav': np.asarray(uav_coords, dtype=np.flaot32)}
        return obs

    def get_ooi_and_mask(self):
        ooi = self.get_ooi()
        mask = self.get_ooi_mask(ooi)
        ooi = self.pad_ooi(ooi)
        return ooi, mask

    def get_ooi(self):
        return self.active_oois

    def pad_ooi(self, ooi):
        fill_obj = (1, 1)
        fil_objs = [fill_obj for _ in range(self.max_ooi - len(ooi))]
        ooi.extend(fil_objs)
        return ooi

    def get_ooi_mask(self, ooi):
        ones = np.ones((len(ooi),))
        zeros = np.zeros((self.max_ooi - len(ooi),))
        mask = np.concatenate([ones, zeros])
        return mask

    def get_uav_coords(self):
        state_arr_c = self.testpp.getState_c(self.test)
        plan = DroneState.from_address(state_arr_c + 0)
        # rl = DroneState.from_address(state_arr_c + 40)
        d1 = DroneState.from_address(state_arr_c + 80)
        d2 = DroneState.from_address(state_arr_c + 120)
        uav_coords = list()
        for uav in [plan, d1, d2]:
            y = uav.pos.y
            x = uav.pos.x
            course = uav.norm_course
            rad_course = np.deg2rad(course)
            sin, cos = np.sin(rad_course), np.cos(rad_course)
            uav_coords.append([y, x, sin, cos])
        return uav_coords

    def get_latest_plan_ooi(self):  # todo: проверить правильность порядка координат
        n = ctypes.c_int(self.plan_n_detected)
        obj_p = self.testpp.getObjectState_c(self.test, n)
        obj = Pos2D.from_address(obj_p)
        obj = np.asarray([obj.y, obj.x])
        return obj

    def get_last_surveyed_ooi(self, uav_index):
        if uav_index == 2:
            n = ctypes.c_int(self.d1_n_detected - 1)
        else:
            n = ctypes.c_int(self.d2_n_detected - 1)
        obj_p = self.testpp.getObjectState_SIR_STV_с(self.test, n, ctypes.c_int(uav_index))
        obj = Pos2D.from_address(obj_p)
        obj = np.asarray([obj.y, obj.x])
        return obj

    def if_active_ooi(self, ooi):
        for point in self.active_oois:
            if np.sqrt(np.sum((ooi - point) ** 2)) < self.min_dist/4:
                return True, point
        else:
            return False, None

    def remove_active_ooi(self, ooi):
        self.active_oois.remove(ooi)
