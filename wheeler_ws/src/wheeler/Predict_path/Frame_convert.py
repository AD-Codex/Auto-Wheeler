
import numpy as np

# -------------------------- convert to rbot frame coords --------------------
# 0 = robot orientation
# rotation_matrix = |  cos( 0), sin( 0) |
#                   | -sin( 0), cos( 0) |

# convert coordinate to robot frame
def Convert_To_Robot_Frame( init_state, ref_state_val):

    ref_state_val = ref_state_val - init_state
    rotation_matrix = np.array([[ np.cos(init_state[2][0]), np.sin(init_state[2][0])],
                                [ -np.sin(init_state[2][0]),  np.cos(init_state[2][0])]])
    
    for i in range( len(ref_state_val[0])):
        rotated_coord = rotation_matrix @ ref_state_val[:2, i]
        ref_state_val[:2, i] = rotated_coord

    init_state = np.array([[0], [0], [0] ])

    # print(ref_state_val)

    return init_state, ref_state_val


def Convert_Obj_To_Robot_Frame( init_state, objStates):
    rotation_matrix = np.array([[ np.cos(init_state[2][0]), np.sin(init_state[2][0])],
                                [ -np.sin(init_state[2][0]),  np.cos(init_state[2][0])]])
    
    for i in range( len(objStates[0])):
        objStates[:2, i] = objStates[:2, i] - init_state[:2, 0]
        objStates[:2,i] = rotation_matrix @ objStates[:2,i]

    return objStates

def Convert_To_Robot_Frame2( init_state, ref_state_val, obj_state):
    for i in range( len(ref_state_val[0])):
        ref_state_val[0][i] = ref_state_val[0][i] - init_state[0][0]
        ref_state_val[1][i] = ref_state_val[1][i] - init_state[1][0]
        ref_state_val[2][i] = ref_state_val[2][i] - init_state[2][0]


    rotation_matrix = np.array([[ np.cos(init_state[2][0]), np.sin(init_state[2][0])],
                                [ -np.sin(init_state[2][0]),  np.cos(init_state[2][0])]])
    
    for i in range( len(ref_state_val[0])):
        rotated_coord = rotation_matrix @ ref_state_val[:2, i]
        ref_state_val[:2, i] = rotated_coord

    for i in range( len(obj_state[0])):
        obj_state[0][i] = obj_state[0][i] - init_state[0][0]
        obj_state[1][i] = obj_state[1][i] - init_state[1][0]

        obj_state[0:2,i] = rotation_matrix @ obj_state[0:2,i]

    init_state = np.array([[0], [0], [0] ])

    return init_state, ref_state_val, obj_state



# --------------------------- convert to world frame coords ---------------------
# rotation_matrix = |  cos( 0),-sin( 0) |
#                   |  sin( 0), cos( 0) |
def Convert_To_World_Frame( init_state, ref_state_val):

    rotation_matrix = np.array([[ np.cos(init_state[2][0]), -np.sin(init_state[2][0])],
                                [ np.sin(init_state[2][0]),  np.cos(init_state[2][0])]])
    
    for i in range( len(ref_state_val[0])):
        rotated_coord = rotation_matrix @ ref_state_val[:2, i]
        ref_state_val[:2, i] = rotated_coord

    ref_state_val = ref_state_val + init_state

    
    return init_state, ref_state_val



def Convert_To_World_Frame2( init_state, ref_state_val, obj_state):

    rotation_matrix = np.array([[ np.cos(init_state[2][0]), -np.sin(init_state[2][0])],
                                [ np.sin(init_state[2][0]),  np.cos(init_state[2][0])]])
    
    for i in range( len(ref_state_val[0])):
        rotated_coord = rotation_matrix @ ref_state_val[:2, i]
        ref_state_val[:2, i] = rotated_coord

    ref_state_val = ref_state_val + init_state

    for i in range( len(obj_state[0])):
        obj_state[0][i] = obj_state[0][i] + init_state[0][0]
        obj_state[1][i] = obj_state[1][i] + init_state[1][0]
    

    return init_state, ref_state_val, obj_state



# ref_state_val = np.array([[0, 0.05,  0.1, 0.15,  0.2, 0.25,  0.3, 0.35,  0.4, 0.45,  0.5, 0.55,  0.6, 0.65,  0.7, 0.75], 
#                           [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0], 
#                           [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0]])

# a, b = Convert_To_Robot_Frame( np.array([[1], [1], [0.1] ]) , ref_state_val)
# print(b)
# c , d = Convert_To_World_Frame( np.array([[1], [1], [0.1] ]) , b)
# print(d.round(decimals=2))