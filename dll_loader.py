import ctypes
from ctypes import *


n_objs = 10

class struct_1(ctypes.Structure):
    _fields_ = [('val1', ctypes.c_int),
                ('val2', ctypes.c_int),
                ('val3', ctypes.c_int),
                ('val4', ctypes.c_int)]


class Pos2D(ctypes.Structure):
    _fields_ = [('x', ctypes.c_double),
                ('y', ctypes.c_double)]


class struct_2(ctypes.Structure):
    _fields_ = [('pos', Pos2D),
                ('h', ctypes.c_double),
                ('course', ctypes.c_double),
                ('norm_course', ctypes.c_double)]


testpp = cdll.LoadLibrary("../dll/Aurora__1_model_solution.dll")
testpp.export_class_c.restype = ctypes.c_void_p
test = testpp.export_class_c()

path = b"C:\\\\prjs\\\\dll\\\\"
path = list(path)
chars = [c_char(x) for x in path]
char_m = c_char * len(path)
path_c = char_m(*chars)

testpp.init.argtypes = [ctypes.c_void_p, POINTER(char_m)]
testpp.step_c.argtypes = [ctypes.c_void_p]
testpp.reset_c.argtypes = [ctypes.c_void_p]
testpp.getState_c.argtypes = [ctypes.c_void_p]
testpp.getVisionState_c.argtypes = [ctypes.c_void_p]
testpp.getObjectState_c.argtypes = [ctypes.c_void_p, ctypes.c_int]

testpp.step_c.restype = ctypes.c_void_p
testpp.reset_c.restype = ctypes.c_void_p
testpp.getState_c.restype = ctypes.c_void_p
testpp.getVisionState_c.restype = ctypes.c_void_p
testpp.getObjectState_c.restype = ctypes.c_void_p

testpp.init(test, path_c)
reset_ret = testpp.reset_c(test)
print(reset_ret)
step_p = testpp.step_c(test)


pos_arr = Pos2D * 4
course = Pos2D(-1, -1)
arr = [course for _ in range(4)]
arr_c = pos_arr(*arr)
testpp.setCourse_c.argtypes = [ctypes.c_void_p, POINTER(pos_arr)]
testpp.setCourse_c(test, arr_c)
testpp.setVisionCourse_c.argtypes = [ctypes.c_void_p, POINTER(pos_arr)]
testpp.setVisionCourse_c(test, arr_c)

nn = 0
uav1, uav2, uav3, uav4 = list(), list(), list(), list()
num_steps = 0
doobsled = 0

while True:
    num_steps += 1
    step_p = testpp.step_c(test)
    step_ret = struct_1.from_address(step_p)
    #uav1.append(step_ret.val1); uav2.append(step_ret.val2); uav3.append(step_ret.val3); uav4.append(step_ret.val4)

    if step_ret.val1 == 1:
        nn += 1
        int_num = c_int(nn)
        ret_obj_state = testpp.getObjectState_c(test, int_num)
        obj_state = Pos2D.from_address(ret_obj_state+40)

        with open("new_detect_list.txt", 'a') as f:
            f.write(str(obj_state.y*100) + "   " + str(obj_state.x*100) + "\n")


    state_p = testpp.getState_c(test)

    if step_ret.val3 == 4:
        doobsled += 1
        with open("drone3_detect.txt", 'a') as f:
            state_ret = struct_2.from_address(state_p + 80)
            f.write(str(state_ret.pos.y) + ';' + str(state_ret.pos.x) + '\n')

    if step_ret.val4 == 4:
        pass
        with open("drone4_detect.txt", 'a') as f:
            state_ret = struct_2.from_address(state_p + 120)
            f.write(str(state_ret.pos.y) + ';' + str(state_ret.pos.x) + '\n')

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


    if nn == 10:
        break


    #print("h: %f\tcourse: %f\tnorm_course: %f\tx: %f\ty: %f" % (state_ret.h, state_ret.course,
    #                                                            state_ret.norm_course, state_ret.pos.x,
    #                                                            state_ret.pos.y))
print(doobsled)