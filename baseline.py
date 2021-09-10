import numpy as np
from functools import partial
distance_vectorizer = partial(np.vectorize, signature='(n),(m)->()')


class BaselineController:
    def __init__(self, distance_mode="l1", turn_radius=1):
        assert distance_mode in ["l1", "turn"], "Invalid distance mode. Must be \"l1\" or \"turn\""
        self.distance_mode = distance_mode
        self.turn_radius = turn_radius

    def __call__(self, observation):
        ooi = observation['ooi_coords']
        ooi_mask = observation['ooi_mask']
        uav_coords = observation['uav']

        active_oois = [ooi[i] for i in range(ooi.shape[0]) if ooi_mask[i] == 1]
        plan_uav = uav_coords[0]
        d1_uav = uav_coords[1]
        d2_uav = uav_coords[2]

        if not len(active_oois):
            d1_course = plan_uav[:2]
            d2_course = plan_uav[:2]

        elif len(active_oois) == 1:
            nearest_uav_index = self.get_nearest_uav_index(d1_uav, d2_uav, active_oois[0])
            if nearest_uav_index == 1:
                d1_course = active_oois[0]
                d2_course = plan_uav[:2]
            else:
                d1_course = plan_uav[:2]
                d2_course = active_oois[0]

        else:
            d1_course, d2_course = self.get_course_to_nearest_points(d1_uav, d2_uav, active_oois)

        action = np.concatenate([d1_course, d2_course])
        return action

    def get_nearest_uav_index(self, d1_uav, d2_uav, ooi):
        d1_distance = self.uav2ooi_distance(d1_uav, ooi)
        d2_distance = self.uav2ooi_distance(d2_uav, ooi)
        if d1_distance < d2_distance:
            index = 1
        else:
            index = 2
        return index

    def get_course_to_nearest_points(self, d1_uav, d2_uav, active_oois):
        d1_distances = self.uav2ooi_distance(d1_uav, active_oois)
        d2_distances = self.uav2ooi_distance(d2_uav, active_oois)

        d1_nearest_index = np.argmin(d1_distances)
        d2_nearest_index = np.argmin(d2_distances)

        if d1_nearest_index == d2_nearest_index:
            if d1_distances[d1_nearest_index] > d2_distances[d2_nearest_index]:
                d1_nearest_index = np.argsort(d1_distances)[1]
                d1_course = active_oois[d1_nearest_index]
                d2_course = active_oois[d2_nearest_index]
            else:
                d1_course = active_oois[d1_nearest_index]
                d2_nearest_index = np.argsort(d2_distances)[1]
                d2_course = active_oois[d2_nearest_index]

        else:
            d1_course = active_oois[d1_nearest_index]
            d2_course = active_oois[d2_nearest_index]

        return d1_course, d2_course

    @distance_vectorizer
    def uav2ooi_distance(self, uav_coord, ooi_coord):
        if self.distance_mode == "l1":
            distance = self.norm_distance(uav_coord, ooi_coord)
        elif self.distance_mode == "turn":
            distance = self.turn_distance(uav_coord, ooi_coord)
        else:
            distance = None
        return distance

    @staticmethod
    def norm_distance(uav, ooi_coord):
        return np.sqrt(np.sum((uav[:2] - ooi_coord) ** 2))

    def turn_distance(self, uav, ooi_coord):
        return 0
