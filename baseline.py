import numpy as np
from functools import partial
import math
from baseline_options import *
distance_vectorizer = partial(np.vectorize, signature='(n),(m)->()')


class BaselineController:
    def __init__(self, distance_mode="turn", turn_radius=TURN_RADIUS):
        assert distance_mode in ["l1", "turn"], "Invalid distance mode. Must be \"l1\" or \"turn\""
        self.distance_mode = distance_mode
        self.turn_radius = turn_radius

    def __call__(self, observation):
        ooi = observation['ooi_coords']
        ooi = ooi[:, ::-1]
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
        d1_distances, d2_distances = list(), list()
        for ooi in active_oois:
            d1_distances.append(self.uav2ooi_distance(d1_uav, ooi))
            d2_distances.append(self.uav2ooi_distance(d2_uav, ooi))
        d1_distances, d2_distances = np.array(d1_distances), np.array(d2_distances)

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

    @staticmethod
    def get_intersections(x0, y0, r0, x1, y1, r1):
        # circle 1: (x0, y0), radius r0
        # circle 2: (x1, y1), radius r1

        d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # non intersecting
        if d > r0 + r1:
            return {}
        # One circle within other
        if d < abs(r0 - r1):
            return {}
        # coincident circles
        if d == 0 and r0 == r1:
            return {}
        else:
            a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
            h = math.sqrt(r0 ** 2 - a ** 2)
            x2 = x0 + a * (x1 - x0) / d
            y2 = y0 + a * (y1 - y0) / d
            x3 = x2 + h * (y1 - y0) / d
            y3 = y2 - h * (x1 - x0) / d
            x4 = x2 - h * (y1 - y0) / d
            y4 = y2 + h * (x1 - x0) / d
            return x3, y3, x4, y4

    def turn_distance(self, uav, ooi_coord):
        y, x, sin, cos = uav

        ooi_y, ooi_x = ooi_coord
        ooi_y, ooi_x = ooi_y - y, ooi_x - x
        y, x = 0, 0

        course_x, course_y = sin, cos
        norm_course_x, norm_course_y = cos, -sin

        r1x, r1y = norm_course_x * self.turn_radius, norm_course_y * self.turn_radius
        r2x, r2y = -r1x, -r1y
        if r1x*ooi_x+r1y*ooi_y > 0:
            rx, ry = r1x, r1y
        else:
            rx, ry = r2x, r2y

        a = ((rx - ooi_x)**2 + (ry - ooi_y)**2)
        if a - self.turn_radius**2 < 0:
            return float("inf")

        hord = np.sqrt(a - self.turn_radius**2)

        _ox, _oy = ooi_x - rx, ooi_y - ry
        _z = -_oy/_ox
        _lam = - (hord**2 - self.turn_radius**2 - _oy**2 - _ox**2) / (2 * _ox)
        _a = 1 + _z**2
        _b = 2 * _z * _lam
        _c = _lam**2 - self.turn_radius**2
        _D = _b**2 - 4 * _a * _c

        _y1, _y2 = (- _b + np.sqrt(_D)) / (2 * _a), (- _b - np.sqrt(_D)) / (2 * _a)
        _x1, _x2 = _z * _y1 + _lam, _z * _y2 + _lam

        quoters = list()
        for x, y in [(_x1, _y1), (_x2, _y2)]:
            if x <= 0:
                if y >= 0:
                    quoters.append(1)
                else:
                    quoters.append(4)
            else:
                if y >= 0:
                    quoters.append(2)
                else:
                    quoters.append(3)

        if quoters[0] != quoters[1]:
            if quoters[0] > quoters[1]:
                turn_x, turn_y = _x2 + rx, _y2 + ry
                quoter = quoters[1]
            else:
                turn_x, turn_y = _x1 + rx, _y1 + ry
                quoter = quoters[0]

        else:
            _vec = (_x2 - _x1, _y2 - _y1)
            s = course_x * _vec[0] + course_y * _vec[1]

            if quoters[0] in [1, 3]:
                if s > 0:
                    turn_x, turn_y = _x1 + rx, _y1 + ry
                    quoter = quoters[0]
                else:
                    turn_x, turn_y = _x2 + rx, _y2 + ry
                    quoter = quoters[1]
            else:
                if s > 0:
                    turn_x, turn_y = _x2 + rx, _y2 + ry
                    quoter = quoters[1]
                else:
                    turn_x, turn_y = _x1 + rx, _y1 + ry
                    quoter = quoters[0]

        _a = np.sqrt(turn_x**2 + turn_y**2)
        cos_a = (_a**2 - 2 * self.turn_radius**2) / (-2*self.turn_radius*self.turn_radius)
        angle = np.arccos(cos_a)
        if quoter in [3, 4]:
            angle = 2*np.pi - angle
        turn_dist = angle*self.turn_radius

        return hord + turn_dist
