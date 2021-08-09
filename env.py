import ctypes
from ctypes import *
import gym
from gym import spaces
import numpy as np
import pandas

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


course_arr = Pos2D * 2


class RandomPointsGenerator:
    @staticmethod
    def baseline_place_points(n_regions, per_region_points, per_std, w=1, h=1):
        points = []
        for _ in range(n_regions):
            n_points = np.round(np.random.uniform(per_region_points[0], per_region_points[1], 1)).astype(np.int32)[0]
            std = np.random.uniform(per_std[0], per_std[1], n_points)[0]
            cx, cy = np.random.uniform(0, 1, 2)
            cx, cy = cx * w, cy * h

            xx = np.random.normal(scale=std, size=n_points) + cx
            yy = np.random.normal(scale=std, size=n_points) + cy

            for x, y in zip(xx, yy):
                points.append([x, y])

        return np.array(points)

    @staticmethod
    def random_over_random():
        n = np.random.randint(1, 5)
        np_min = 5 + np.random.randint(0, 20)
        np_max = np_min + np.random.randint(0, 25 - np_min)
        s_min = 0.05 + np.random.rand() * 0.15
        s_max = s_min + np.random.rand() * (0.2 - s_min)
        return RandomPointsGenerator.baseline_place_points(n, [np_min, np_max], [s_min, s_max], w=1, h=1)

    @staticmethod
    def get_n_random_points(n):
        points = RandomPointsGenerator.random_over_random()
        points = np.random.permutation(points)[:n]
        return points


class Drones(gym.Env):
    def __init__(self, max_ooi=100):
        super(Drones, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.max_ooi = max_ooi
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(max_ooi+2, 4), dtype=np.float32) # todo: заменить observation space
        self.observed_ooi, self.detected_ooi, self.all_ooi, self.cameras_state = list(), list(), list(), list()
        self.testpp = cdll.LoadLibrary("../dll/Aurora__1_model_solution.dll")
        self.test = "uninitialized"
        self.load_library()

    def step(self, action):
        self.set_course(action)
        self.update_cameras()

        #step_events_c = self.testpp.step_c(self.test)
        #events = [step_events.val1, step_events.val2, step_events.val3, step_events.val4]
        events = self.simulation_step()

        if 5 in events:
            # mission finished
            # return reward
            pass

        if events[0] == 1:
            # add new point to the list
            pass

        if 4 in events[2:]:
            # remove point from the list
            pass

        return None

    def reset(self):
        # Reset the state of the environment to an initial state
        pass

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def load_library(self):
        path = b"C:\\\\prjs\\\\dll\\\\"
        path = list(path)
        chars = [c_char(x) for x in path]
        char_m = c_char * len(path)
        path_c = char_m(*chars)

        # define C funcs arg types and return types
        self.testpp.init.argtypes = [ctypes.c_void_p, POINTER(char_m)]
        self.testpp.step_c.argtypes = [ctypes.c_void_p]
        self.testpp.reset_c.argtypes = [ctypes.c_void_p]
        self.testpp.getState_c.argtypes = [ctypes.c_void_p]
        self.testpp.setCourse_c.argtypes = [ctypes.c_void_p, POINTER(course_arr)]
        self.testpp.saveTrackToFile_c.argtypes = [ctypes.c_void_p]

        self.testpp.export_class_c.restype = ctypes.c_void_p
        self.testpp.step_c.restype = ctypes.c_void_p
        self.testpp.reset_c.restype = ctypes.c_void_p
        self.testpp.getState_c.restype = ctypes.c_void_p

        self.test = self.testpp.export_class_c()
        self.testpp.init(self.test, path_c)

    def simulation_step(self):
        step_events_c = self.testpp.step_c(self.test)
        step_events = StepEvents.from_address(step_events_c)
        events = [step_events.val1, step_events.val2, step_events.val3, step_events.val4]
        return events

    def set_course(self, coords):
        uav1_x, uav1_y, uav2_x, uav2_y = coords
        course1, course2 = Pos2D(uav1_x, uav1_y), Pos2D(uav2_x, uav2_y)
        course = [course1, course2]
        course_c = course_arr(*course)
        self.testpp.setCourse_c(self.test, course_c)

    def update_cameras(self):
        pass

    def get_ooi(self):
        pass

    def get_puav_coords(self):
        state_arr_c = self.testpp.getState_c(self.test)
        state = DroneState.from_address(state_arr_c + 0)
        return state
