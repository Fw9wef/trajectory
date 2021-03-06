import ctypes
from ctypes import *
import gym
from gym import spaces
import numpy as np
import pandas as pd
import os
from env_options import *

# кодировка различных событий
EVENTS = {
    0: "nothing",
    1: "new_obs_point",
    2: "waypoint_reached",
    3: "1+2",
    4: "obs_point_searched",
    5: "mc"
}


class StepEvents(ctypes.Structure):
    # структура массива событий для всех бля
    _fields_ = [('val1', ctypes.c_int),
                ('val2', ctypes.c_int),
                ('val3', ctypes.c_int),
                ('val4', ctypes.c_int)]


class Pos2D(ctypes.Structure):
    # структура двухмерных координат
    _fields_ = [('x', ctypes.c_double),
                ('y', ctypes.c_double)]


class DroneState(ctypes.Structure):
    # структура положения и направления движения дрона
    _fields_ = [('pos', Pos2D),
                ('h', ctypes.c_double),
                ('course', ctypes.c_double),
                ('norm_course', ctypes.c_double)]


course_arr = Pos2D * 4


class Angles(ctypes.Structure):
    # структура телесного угла
    _fields_ = [('x', ctypes.c_double),
                ('y', ctypes.c_double),
                ('z', ctypes.c_double)]


class VisionState(ctypes.Structure):
    # структура направления камеры
    _fields_ = [('vertices_pos', course_arr),
                ('angles', Angles)]


class RandomPointsGenerator:
    '''
    Этот класс выполняет расстановку объектов интереса по карте, выполняет расчет курсовых точек для планового и рлс бла
    и записывает полученные данные в 4 .csv файла:
    objects.csv - положение объектов интереса
    start.csv - стартовые координаты бла
    waypoints.csv - курсовые точки планового бла
    waypoints_RL.csv - курсовые точки рлс бла
    '''
    def __init__(self, left_bottom,  # координата левого нижнего угла карты
                 right_top,  # координата правого верхнего угла карты
                 plan_uav_vision_width,  # ширина обзора планового бла в долях карты
                 plan_uav_vision_high,  # высота обзора планового бла в долях карты
                 plan_uav_overlap,  # коэффициент перекрытия зон обзора планового бла при движении змейкой
                 path_to_folder,  # папка, в которую будут записаны .csv файлы
                 x100=True,  # координаты точек интереса должны быть в 100 раз больше чем
                             # соответствующие координаты курсовых точек бла (это такой баг)
                 debug=DEBUG  # включение режима дебага
                 ):
        if x100:
            left_bottom = [x * 100 for x in left_bottom]
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
        self.debug = debug

    def __call__(self, n_range, read=False):
        '''
        При вызове объект делает все то, что было в описании класса выше.
        n_range: [min, max] - интервал количества объектов интереса на карте
        read: bool - считать всю необходимую информацию из файлов вместо генерации и записи
        '''
        if read:
            self.objects = pd.read_csv(os.path.join(self.path_to_folder, "objects.csv"), sep=';').values[:,:2]
            self.plan_waypoints = pd.read_csv(os.path.join(self.path_to_folder, "waypoints.csv"), sep=';').values[:,:2]
            self.start_points = pd.read_csv(os.path.join(self.path_to_folder, "start.csv"), sep=';').values[:,:2]
            return {
                'objs': self.objects,
                'plan_waypoints': self.plan_uav_waypoints,
                'start_points': self.start_points
            }
        else:
            self.get_n_random_points(n_range)
            if self.debug:
                print("ooi points generated")
            self.get_start_positions()
            if self.debug:
                print("start points generated")
            self.get_plan_uav_waypoints()
            if self.debug:
                print("plan uav waypoints generated")
            self.get_rl_uav_waypoint()
            if self.debug:
                print("rl uaw waypoints generated")
            self.normalize_coordinates()
            if self.debug:
                print("coordinates normolized")
            self.save_start_data()
            if self.debug:
                print("start data saved")
            return {
                'objs': self.objects,
                'plan_waypoints': self.plan_uav_waypoints,
                'start_points': self.start_points
            }

    @staticmethod
    def place_points(n_regions, per_region_points, per_std, w=1, h=1):
        '''
        Метод расставляет на карте n_regions гауссианов, в каждом из которых per_region_points точек
        с per_std стандартным отклонением
        :param n_regions: int - количество гауссианов
        :param per_region_points: [int, int] - интервал количества точек в каждом гауссиане
        :param per_std: [float, float] - интервал стандартного отклонения для каждого гауссиана
        :param w: float - нормировка по ширине
        :param h: float - нормировка по высоте
        :return: np.array shape=(n, 2) - сгенерированные точки
        '''
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

        objects_t = list()
        for obj in objects:
            if 0.05 < obj[0] < 0.95 and 0.05 < obj[1] < 0.95:
                objects_t.append(obj)

        return np.array(objects_t)

    @staticmethod
    def random_over_random():
        '''
        Функция генерирует число гауссианов и верхние и нижние граници интервалов для метода place_points
        :return: np.array shape=(n, 2) - сгенерированные точки
        '''
        n = np.random.randint(1, 5)
        np_min = 5 + np.random.randint(0, 20)
        np_max = np_min + np.random.randint(0, 25 - np_min)
        s_min = 0.05 + np.random.rand() * 0.15
        s_max = s_min + np.random.rand() * (0.2 - s_min)
        return RandomPointsGenerator.place_points(n, [np_min, np_max], [s_min, s_max], w=1, h=1)

    def get_n_random_points(self, n_range):
        '''
        Функция генерирует точки интереса и выбирает из всех сгенерированных не более n штук,
        где n - число из интервала n_range.
        Сгенерированные и отобранные точкисохраняются в атрибуте объекта класса objects
        :param n_range: [int, int] - интервал максимального количества точек
        :return: None
        '''
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
        '''
        По сгенерированным точкам и начальной точке прокладывает маршрут для планового бла.
        Информация сохраняется в атрибуте объекта класса plan_uav_waypoints (list)
        :return: None
        '''

        # сначала определяем границы минимального прямоугольника, содержащего все точки интереса
        left, right, bot, top = np.inf, -np.inf, np.inf, -np.inf
        for y, x in self.objects:
            left = x if x < left else left
            right = x if x > right else right
            bot = y if y < bot else bot
            top = y if y > top else top

        # задаем длину движения по высоте и ширине
        vertical_run_l = top - bot
        horizontal_run_l = 2*self.plan_uav_vision_width - self.plan_uav_overlap

        start_y, start_x = self.start_points['plan']
        waypoints = list()
        movement_list = list()

        # сначала летим к ближайшему углу найденного прямоугольника
        # от угла зависит куда дальше полетит бла: вверх или вниз, вправо или влево
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

        # состовляем минимальный цикл движения в прямоугольнике
        if movement_list[0] == 'go_top':
            movement_list.append('go_bot')
        else:
            movement_list.append('go_top')

        if movement_list[1] == 'go_right':
            movement_list.append('go_right')
        else:
            movement_list.append('go_left')

        waypoints.append((curr_waypoint_y, curr_waypoint_x))
        if self.debug:
            print(stop_condition)
        if stop_condition == 'right':
            condition = lambda inp: inp < right
            if self.debug:
                print(right)
        elif stop_condition == 'left':
            condition = lambda inp: inp > left
            if self.debug:
                print(left)

        # овторяем цикл пока не пролетим весь прямоугольник
        n_step = 0
        while True:
            if self.debug:
                print(curr_waypoint_x)
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
            n_step += 1
            if (not condition(curr_waypoint_x)) and (movement in ['go_top', 'go_bot']):
                break

        exit_x = 0 if curr_waypoint_x < 0.5 else 1
        exit_y = 0 if curr_waypoint_y < 0.5 else 1
        waypoints.append((exit_y, exit_x))
        self.plan_uav_waypoints = waypoints

    def get_rl_uav_waypoint(self):
        '''
        Прокладывает маршрут рлс бла.
        Информация сохраняется в атрибуте объекта класса rl_uav_waypoints (list)
        :return: None
        '''
        start_y, start_x = self.start_points['rl']
        corner_x = 0.1 if start_x < 0.5 else 0.9
        corner_y = 0.1 if start_y < 0.5 else 0.9
        exit_corner_x = 0.1 if corner_x > 0.5 else 0.9
        waypoints = [(corner_y, corner_x), (corner_y, exit_corner_x)]
        self.rl_uav_waypoints = waypoints

    def get_start_positions(self):
        '''
        Функция расставляет стартовые позиции всех бла.
        Информация сохраняется в атрибуте объекта класса start_points (dict)
        :return: None
        '''
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
        '''
        Переводим координаты точек интереса и курсовых точек планового и рлс бла от долей карты в декартовы координаты
        :return: None
        '''
        self.objects = np.asarray(self.objects) * self.delta + self.bias
        self.plan_uav_waypoints = np.asarray(self.plan_uav_waypoints) * self.delta + self.bias
        self.rl_uav_waypoints = np.asarray(self.rl_uav_waypoints) * self.delta + self.bias
        for key, value in self.start_points.items():
            self.start_points[key] = value * self.delta + self.bias

    def save_start_data(self):
        '''
        Сохраняем все полученные координаты точек интереса и курсовых точек
        :return: None
        '''
        self.save_objects()
        self.save_plan_waypoints()
        self.save_rl_waypoints()
        self.save_start_positions()

    def save_objects(self):
        '''
        Сохраняем объекты интереса
        :return: None
        '''
        data = {
            'Latitude': self.objects[:, 0],
            'Longitude': self.objects[:, 1],
            'Altitude': [H for _ in range(self.objects.shape[0])],
            'class': ['car' for _ in range(self.objects.shape[0])]
        }
        data = pd.DataFrame(data)
        path_to_csv = os.path.join(self.path_to_folder, "objects.csv")
        data.to_csv(path_to_csv, index=False, sep=';')

    def save_plan_waypoints(self):
        '''
        Сохраняем курсовые точки планового бла
        :return: None
        '''
        data = {
            'Latitude': self.plan_uav_waypoints[:, 0],
            'Longitude': self.plan_uav_waypoints[:, 1],
            'Altitude': [H for _ in range(self.plan_uav_waypoints.shape[0])],
        }
        data = pd.DataFrame(data)
        path_to_csv = os.path.join(self.path_to_folder, "waypoints.csv")
        data.to_csv(path_to_csv, index=False, sep=';')

    def save_rl_waypoints(self):
        '''
        Сохраняем курсовые точки рлс бла
        :return: None
        '''
        data = {
            'Latitude': self.rl_uav_waypoints[:, 0],
            'Longitude': self.rl_uav_waypoints[:, 1],
            'Altitude': [H for _ in range(self.rl_uav_waypoints.shape[0])],
        }
        data = pd.DataFrame(data)
        path_to_csv = os.path.join(self.path_to_folder, "waypoints_RL.csv")
        data.to_csv(path_to_csv, index=False, sep=';')

    def save_start_positions(self):
        '''
        Сохраняем стартовые позиции бла
        :return: None
        '''
        start_points = list()
        for key in ['plan', 'rl', 'd1', 'd2']:
            start_points.append(self.start_points[key])
        start_points = np.asarray(start_points)
        data = {
            'Latitude': start_points[:, 0],
            'Longitude': start_points[:, 1],
            'Altitude': [H for _ in range(start_points.shape[0])],
        }
        data = pd.DataFrame(data)
        path_to_csv = os.path.join(self.path_to_folder, "start.csv")
        data.to_csv(path_to_csv, index=False, sep=';')


class Drones(gym.Env):
    '''
    Этот класс представляет из себя python обертку среды.
    Для совместимости с большинством существующих rl-библиотек она наследует класс gym.Env.
    '''
    def __init__(self, work_dir=WORK_DIR, max_steps=MAX_STEPS, n_range=N_RANGE,
                 agent_index=AGENT_INDEX, time_penalty=TIME_PENALTY, survey_reward=SURVEY_REWARD,
                 mc_reward=MC_REWARD, left_bottom=LEFT_BOTTOM, right_top=RIGHT_TOP,
                 plan_uav_vision_width=PLAN_UAV_VISION_WIDTH, plan_uav_vision_high=PLAN_UAV_VISION_HIGH,
                 plan_uav_overlap=PLAN_UAV_OVERLAP, debug=DEBUG, baseline=BASELINE,
                 flight_delay=FLIGHT_DELAY, delay_steps=DELAY_STEPS):
        super(Drones, self).__init__()
        self.max_ooi = n_range[1]
        self.max_steps = max_steps
        self.n_range = n_range
        self.time_penalty = time_penalty
        self.survey_reward = survey_reward
        self.mc_reward = mc_reward

        # Задаем пространство действий и наблюдений
        # Это должны быть объекты типа gym.spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict({'ooi_coords': spaces.Box(-1, 1, (n_range[1], 2), np.float32),
                                              'ooi_mask': spaces.Box(0, 1, (n_range[1],)),
                                              'uav': spaces.Box(-1, 1, (3, 4), np.float32)})

        self.work_dir = work_dir
        self.agent_index = str(agent_index)
        self.agent_dir = os.path.join(self.work_dir, self.agent_index)
        if not os.path.isdir(self.agent_dir):
            os.mkdir(self.agent_dir)
        b_agent_dir = bytes(self.agent_dir, encoding="utf-8")
        b_agent_dir = list(b_agent_dir)
        chars = [c_char(x) for x in b_agent_dir]
        self.char_m = c_char * len(b_agent_dir)
        self.agent_dir_c = self.char_m(*chars)
        self.left_bottom, self.right_top = np.asarray(left_bottom), np.asarray(right_top)
        self.high, self.width = self.right_top - self.left_bottom
        self.exp_init = RandomPointsGenerator(left_bottom, right_top, plan_uav_vision_width,
                                              plan_uav_vision_high, plan_uav_overlap,
                                              path_to_folder=self.agent_dir, debug=debug)

        # тут выполняется загрузка dll
        self.testpp = cdll.LoadLibrary(PATH_TO_DLL)
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

        self.flight_delay = flight_delay
        self.delay_steps = delay_steps
        self.debug = debug
        self.baseline = baseline

    def step(self, action):
        '''
        Это основной метод для работы со средой. Он позволяет указать курсовые точки для бла дообследования
        и получить обратную связь в виде нового наблюдения среды и награды.
        :param action: array-like shape=(4,) - координаты курсовых точек для бла
        :return: obs - наблюдение среды (см. self.observation_space)
                 step_reward: float - награда за предыдущий шаг
                 done: bool - закончен ли эпизод
                 message: dict - дополнительная информация. Не используется в обучении, но позволяет вытаскивать
                                 больше информации из цикла
        '''
        step_reward = 0.
        self.set_course(action)
        message = dict()

        break_flag = False
        message["need_new_action"] = False

        # Сеть совершает новое действие только при достижении курсовых точек или обследовании новых точек интереса.
        while True:
            self.total_steps += 1
            step_reward -= self.time_penalty

            if self.total_steps in self.delay_steps:
                self.set_course(action)

            self.update_cameras()
            events = self.simulation_step()

            # new active ooi
            if events[0] == 1:
                self.plan_n_detected += 1
                new_ooi = self.get_latest_plan_ooi()
                self.active_oois.add(new_ooi)
                break_flag = True
                message["plan"] = new_ooi
                message["need_new_action"] = True

            # waypoint reached
            if 2 in events[2:]:
                break_flag = True
                message["need_new_action"] = True

            # ooi surveyed
            for i, e in enumerate(events[2:], start=2):
                if e == 4:

                    if i == 2:
                        self.d1_n_detected += 1
                    elif i == 3:
                        self.d2_n_detected += 1

                    unique, gt_ooi = self.if_active_ooi(i)
                    if unique:
                        message[i] = tuple(gt_ooi)
                        self.remove_active_ooi(gt_ooi)
                        break_flag = True
                        message["need_new_action"] = True
                        step_reward += self.survey_reward
                        self.surveyed_oois += 1

            # возвращаем положение камер через message
            if self.debug:
                message["cams"] = self.get_cams()
                break_flag = True

            # Выход из цикла при достижении максимального шага среды
            if self.total_steps > MAX_STEPS:
                break_flag = True

            if break_flag:
                break

        # check for mission complete
        if self.surveyed_oois == len(self.oois):
            done = True
            step_reward += self.mc_reward
        elif self.total_steps > MAX_STEPS:
            done = True
        else:
            done = False

        # берем следующее наблюдение
        obs = self._next_observation()

        return obs, step_reward, done, message

    def reset(self):
        '''
        Метод reset устанавливает среду в начальное положение.
        Запускает генератор точек, сбрасывает значение атрибутов.
        :return: obs - наблюдение среды
        '''
        exp_data = self.exp_init(self.n_range)
        self.testpp.init_c(self.test, self.agent_dir_c)
        self.testpp.reset_c(self.test)
        self.oois = exp_data['objs']
        self.min_dist = self.find_min_dist_btw_oois()
        self.plan_waypoints = exp_data['plan_waypoints']
        self.active_oois = set()
        self.total_steps = 0
        self.surveyed_oois = 0
        self.plan_n_detected = 0
        self.d1_n_detected = 0
        self.d2_n_detected = 0
        return self._next_observation()

    def render(self, mode='human', close=False):
        '''
        Заглушка.
        Нужна для обеспечения совместимости с библиотеками.
        '''
        pass

    def close(self):
        '''
        Закрытие среды
        :return: None
        '''
        self.testpp.destruct_c(self.test)

    def load_library(self):
        '''
        Определяем типы принимаемых и возвращаемых переменных для C-шных функций из dll
        :return: None
        '''
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
        '''
        Поиск минимального растояния между всеми парами объектов интереса
        :return: float - минимальное расстояние между парой объектов интереса
        '''
        min_dist = np.inf
        for i in range(len(self.oois)):
            for j in range(i+1, len(self.oois)):
                x, y = self.oois[i], self.oois[j]
                dist = np.sqrt(np.sum((x - y) ** 2))
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def simulation_step(self):
        '''
        Обертка для функции шага среды
        :return: events (list) - события произошедшие с каждым бла (ничего, долетел до курсовой точки,
                                                                    обследовал точку интереса,...)
        '''
        step_events_c = self.testpp.step_c(self.test)
        step_events = StepEvents.from_address(step_events_c)
        events = [step_events.val1, step_events.val2, step_events.val3, step_events.val4]
        return events

    def norm_coords(self, y, x):
        '''
        Нормировка координат. Сеть выдает координаты в диапазоне (-1, 1) - такие необходимо
        привести в декартовы координаты на карте (от левой нижней точки до правой верхней).
        Baseline метод выдает сразу декартовы координаты - их преобразовывать не надо.
        :param y: float - координата y
        :param x: float - координата x
        :return: (float, float) - преобразованная пара координат
        '''
        if not self.baseline:
            y = self.left_bottom[1] + self.width * (y + 1) / 2
            x = self.left_bottom[0] + self.high * (x + 1) / 2
        return y, x

    def set_course(self, coords):
        '''
        Обертка функции для передачи курсовых координат бла.
        :param coords: (float, float, float, float) - по паре координат на бла
        :return: None
        '''
        uav1_y, uav1_x, uav2_y, uav2_x = coords
        (uav1_y, uav1_x), (uav2_y, uav2_x) = self.norm_coords(uav1_y, uav1_x), self.norm_coords(uav2_y, uav2_x)
        if self.flight_delay:
            _, uav1, uav2 = self.get_uav_coords()
            if self.total_steps < self.delay_steps[0]:
                uav1_y, uav1_x = uav1[:2]
                uav2_y, uav2_x = uav2[:2]
            elif self.total_steps < self.delay_steps[1]:
                uav2_y, uav2_x = uav2[:2]
        course1, course2 = Pos2D(uav1_x, uav1_y), Pos2D(uav2_x, uav2_y)
        course = [course1, course2, course1, course2]
        course_c = course_arr(*course)
        self.testpp.setCourse_c(self.test, course_c)

    def update_cameras(self):
        '''
        Функция следит за тем, чтобы камеры всегда были направлены на близжайший объект
        :return: None
        '''
        _, uav1, uav2 = self.get_uav_coords()
        objects = self.get_ooi()
        vision_objs = list()
        for num, uav in enumerate([uav1, uav2]):
            obj_coords = self.find_nearest_ooi(uav, objects)
            vision_objs.append(obj_coords)
        self.set_vision_course(vision_objs)

    def find_nearest_ooi(self, uav_coords, objs):
        '''
        Фуункция ищет близжайший к указанным координатам объект интереса из переданного списка
        :param uav_coords: координаты бла
        :param objs: координаты объектов интереса
        :return: (float, float) - координаты близжайшего объекта
        '''
        objs_list = list(objs)
        if len(objs_list):
            uav = np.asarray(uav_coords[:2])
            objs = np.asarray(objs_list)[:, ::-1]
            delta = np.sqrt(np.sum((objs - uav) ** 2, axis=-1))
            nearest_obj_ind = np.argmin(delta)
            nearest_obj = objs_list[nearest_obj_ind][::-1]
        else:
            nearest_obj = (1, 1)
        return nearest_obj

    def set_vision_course(self, vision_objs):
        '''
        Обертка функции для управления камерами бла
        :param vision_objs: - координаты объектов. По одному на бла дообследования
        :return:  None
        '''
        coords1, coords2 = vision_objs
        coords1, coords2 = self.norm_coords(*coords1), self.norm_coords(*coords2)
        coords1, coords2 = Pos2D(coords1[1], coords1[0]), Pos2D(coords2[1], coords2[0])
        coords = [coords1, coords2, coords1, coords2]
        coords_c = course_arr(*coords)
        self.testpp.setVisionCourse_c(self.test, coords_c)

    def _next_observation(self):
        '''
        Функция собирает наблюдение среды: положение известных объектов обследования, маска пэддинга для них
                                           и координаты и направление движения бла
        :return: obs - наблюдение среды
        '''
        ooi_coords, ooi_mask = self.get_ooi_and_mask()
        uav_coords = self.get_uav_coords()
        obs = {'ooi_coords': np.asarray(ooi_coords, dtype=np.float32),
               'ooi_mask': np.asarray(ooi_mask, dtype=np.int8),
               'uav': np.asarray(uav_coords, dtype=np.float32)}
        return obs

    def get_ooi_and_mask(self):
        '''
        Возвращает np.array матрицу размера (self.max_ooi, 2) с известными НЕ дообследованными объектами интереса и
        соответствующую маску пэддингов
        :return:
        '''
        ooi = self.get_ooi()
        mask = self.get_ooi_mask(ooi)
        ooi = self.pad_ooi(ooi)
        return ooi, mask

    def get_ooi(self):
        '''
        Возвращает известные НЕ дообследованные объекты интереса
        :return: np.array - известные НЕ дообследованные объекты интереса
        '''
        return self.active_oois

    def pad_ooi(self, ooi):
        '''
        Добавляет пэддинги матрице известных НЕ дообследованных объектов интереса до размера (self.max_ooi, 2)
        '''
        fill_obj = (1, 1)
        fill_objs = [fill_obj for _ in range(self.max_ooi - len(ooi))]
        ooi = list(ooi)
        ooi.extend(fill_objs)
        return ooi

    def get_ooi_mask(self, ooi):
        '''
        Возвращает маску пэддингов
        '''
        ones = np.ones((len(ooi),))
        zeros = np.zeros((self.max_ooi - len(ooi),))
        mask = np.concatenate([ones, zeros])
        return mask

    def get_uav_coords(self):
        '''
        Обертка dll функции - возвращает координаты планового бла и бла дообследования
        :return: list shape=(3,4) - координаты и направления движения бла
        '''
        state_arr_c = self.testpp.getState_c(self.test)
        plan = DroneState.from_address(state_arr_c + 0)
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

    def get_latest_plan_ooi(self):
        '''
        Возвращает последнюю точку интереса, обследованную плановым бла
        :return: (float, float) - координаты точки
        '''
        n = ctypes.c_int(self.plan_n_detected)
        obj_p = self.testpp.getObjectState_c(self.test, n)
        obj = Pos2D.from_address(obj_p+40)
        obj = (obj.y, obj.x)
        return obj

    def get_last_surveyed_ooi(self, uav_index):
        '''
        Возвращает последнюю точку интереса, обследованную заданным бла дообследования
        :param uav_index: int - номер бла
        :return: (float, float) - координаты точки
        '''
        if uav_index == 2:
            n = ctypes.c_int(self.d1_n_detected)
        else:
            n = ctypes.c_int(self.d2_n_detected)
        obj_p = self.testpp.getObjectState_SIR_STV_с(self.test, n, ctypes.c_int(uav_index))
        obj = Pos2D.from_address(obj_p)
        obj = (obj.x, obj.y)
        return obj

    def if_active_ooi(self, uav_ind):
        '''
        Проверяет дообследованную точку. (Костыль из-за багов с детекцией точек, когда их несколько в поле зрения бла)
        '''
        cams = self.get_cams()
        if uav_ind == 2:
            cam = cams['d1']
        elif uav_ind == 3:
            cam = cams['d2']
        ooi = np.mean(cam, axis=0)[::-1]
        for point in self.active_oois:
            point = np.asarray(point)
            if np.sqrt(np.sum((ooi - point) ** 2)) < 400:
                return True, point
        else:
            return False, None

    def remove_active_ooi(self, ooi):
        '''
        Убирает точку интереса из списка НЕ дообследованных
        :param ooi: координаты точки
        :return: None
        '''
        ooi = (ooi[0], ooi[1])
        self.active_oois.remove(ooi)

    def get_cams(self):
        '''
        Функция возвращает координаты вершин зона обзора для планового бла и бла дообследования.
        :return: np.array shape=(3,8) - координаты
        '''
        cams = dict()
        cams_p = self.testpp.getVisionState_c(self.test)
        for uav, ad_addr in zip(["plan", "d1", "d2"], [0, 88*2, 88*3]):
            cam = VisionState.from_address(cams_p + ad_addr)
            cam_coords = list()
            for i in range(4):
                cam_coords.append((cam.vertices_pos[i].y, cam.vertices_pos[i].x))
            cam_coords.append(cam_coords[0])
            cams[uav] = np.asarray(cam_coords)
        return cams
