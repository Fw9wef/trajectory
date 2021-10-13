import ctypes
from ctypes import *
import numpy as np

n_objs = 10


class struct_1(ctypes.Structure):
    _fields_ = [('val1', ctypes.c_int),
                ('val2', ctypes.c_int),
                ('val3', ctypes.c_int),
                ('val4', ctypes.c_int)]


class Pos2D(ctypes.Structure):
    _fields_ = [('x', ctypes.c_double),
                ('y', ctypes.c_double)]


class Angles(ctypes.Structure):
    _fields_ = [('x', ctypes.c_double),
                ('y', ctypes.c_double),
                ('z', ctypes.c_double)]


VertPos = Pos2D * 4


class VisionState(ctypes.Structure):
    _fields_ = [('vertices_pos', VertPos),
                ('angles', Angles)]


class struct_2(ctypes.Structure):
    _fields_ = [('pos', Pos2D),
                ('h', ctypes.c_double),
                ('course', ctypes.c_double),
                ('norm_course', ctypes.c_double)]


path = b"../trajectory/0"


def qweqwe(path):
    print(path)
    testpp = cdll.LoadLibrary("../dll/Aurora__1_model_solution.dll")
    print(testpp)

    testpp.export_class_c.restype = ctypes.c_void_p
    test = testpp.export_class_c()

    path = list(path)
    chars = [c_char(x) for x in path]
    char_m = c_char * len(path)
    path_c = char_m(*chars)

    testpp.init_c.argtypes = [ctypes.c_void_p, POINTER(char_m)]
    testpp.step_c.argtypes = [ctypes.c_void_p]
    testpp.reset_c.argtypes = [ctypes.c_void_p]
    testpp.getState_c.argtypes = [ctypes.c_void_p]
    testpp.getVisionState_c.argtypes = [ctypes.c_void_p]
    testpp.getObjectState_c.argtypes = [ctypes.c_void_p, ctypes.c_int]
    testpp.getObjectState_SIR_STV_с.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

    testpp.step_c.restype = ctypes.c_void_p
    testpp.reset_c.restype = ctypes.c_void_p
    testpp.getState_c.restype = ctypes.c_void_p
    testpp.getVisionState_c.restype = ctypes.c_void_p
    testpp.getObjectState_c.restype = ctypes.c_void_p
    testpp.getObjectState_SIR_STV_с.restype = ctypes.c_void_p
    print('qweqweqweqweewq')

    testpp.init_c(test, path_c)
    print('init')
    reset_ret = testpp.reset_c(test)
    print('reset')
    step_p = testpp.step_c(test)
    print('init_2')


    pos_arr = Pos2D * 4
    course = Pos2D(100, 100)
    arr = [course for _ in range(4)]
    arr_c = pos_arr(*arr)
    testpp.setCourse_c.argtypes = [ctypes.c_void_p, POINTER(pos_arr)]
    testpp.setCourse_c(test, arr_c)
    testpp.setVisionCourse_c.argtypes = [ctypes.c_void_p, POINTER(pos_arr)]
    testpp.setVisionCourse_c(test, arr_c)

    nn = 0
    uav1, uav2, uav3, uav4 = list(), list(), list(), list()
    num_steps = 0
    doobsled_1 = 0
    doobsled_2 = 0
    save_name = "new_detect_list" + ".txt"
    print('start')
    for _ in range(7000):
        num_steps += 1
        step_p = testpp.step_c(test)
        step_ret = struct_1.from_address(step_p)
        #uav1.append(step_ret.val1); uav2.append(step_ret.val2); uav3.append(step_ret.val3); uav4.append(step_ret.val4)

        if num_steps == 100:
            course = Pos2D(-6645, -4737)
            arr = [course for _ in range(4)]
            arr_c = pos_arr(*arr)
            testpp.setCourse_c.argtypes = [ctypes.c_void_p, POINTER(pos_arr)]
            testpp.setCourse_c(test, arr_c)
            testpp.setVisionCourse_c.argtypes = [ctypes.c_void_p, POINTER(pos_arr)]
            testpp.setVisionCourse_c(test, arr_c)

            ppp = testpp.getVisionState_c(test)
            aaa = VisionState.from_address(ppp)
            print(aaa.vertices_pos[0].x)

        if step_ret.val1 == 1:
            nn += 1
            int_num = c_int(nn)
            ret_obj_state = testpp.getObjectState_c(test, int_num)
            obj_state = Pos2D.from_address(ret_obj_state+40)

            with open(save_name, 'a') as f:
                f.write(str(obj_state.y*100) + "   " + str(obj_state.x*100) + "\n")


        state_p = testpp.getState_c(test)

        if step_ret.val3 == 4:
            doobsled_1 += 1
            dettt = testpp.getObjectState_SIR_STV_с(test, c_int(doobsled_1-1), c_int(2))
            dettt = Pos2D.from_address(dettt)
            print('1'*100)
            print(dettt.x, dettt.y)
            print('1' * 100)
            with open("drone3_detect.txt", 'a') as f:
                state_ret = struct_2.from_address(state_p + 80)
                print(state_ret.course, )
                f.write(str(dettt.y) + ';' + str(dettt.x) + '\n')

        if step_ret.val4 == 4:
            doobsled_2 += 1
            dettt = testpp.getObjectState_SIR_STV_с(test, c_int(doobsled_2-1), c_int(3))
            dettt = Pos2D.from_address(dettt)
            print('2' * 100)
            print(dettt.x, dettt.y)
            print('2' * 100)
            with open("drone4_detect.txt", 'a') as f:
                state_ret = struct_2.from_address(state_p + 120)
                f.write(str(dettt.y) + ';' + str(dettt.x) + '\n')

        with open("drone1.txt", 'a') as f:
            state_ret = struct_2.from_address(state_p + 0)
            f.write(str(state_ret.pos.y) + ';' + str(state_ret.pos.x) + '\n')
        with open("drone2.txt", 'a') as f:
            state_ret = struct_2.from_address(state_p + 40)
            f.write(str(state_ret.pos.y) + ';' + str(state_ret.pos.x) + '\n')
        with open("drone3.txt", 'a') as f:
            state_ret = struct_2.from_address(state_p + 80)
            f.write(str(state_ret.pos.y) + ';' + str(state_ret.pos.x) + '\n')
        with open("drone4.txt", 'a') as f:
            state_ret = struct_2.from_address(state_p + 120)
            f.write(str(state_ret.pos.y) + ';' + str(state_ret.pos.x) + '\n')

    print(doobsled_1)
    print(doobsled_2)


qweqwe(path)

"""
paths = [b"case1", b"case2", b"case3", b"case4"]
paths = [path + x for x in paths]

if __name__ == "__main__":
    from multiprocessing import Process

    procs = [Process(target=qweqwe, args=(p,)) for p in paths]
    for proc in procs:
        proc.start()

    for proc in procs:
        proc.join()
"""

