import daqp
from ctypes import * 
import ctypes.util
import numpy as np
import math




# X_0 = np.array([ [1], [2], [0]])
# dt = 0.1
# dt should be very small

# # reference state values [[ x],[ y],[ theta], [l], [l_k]]
# ref_state_val = np.array([[0, 1, 1.1, 1.2], 
#                           [0, 0, 0, 0], 
#                           [0, 0, 0, 0]])


# # U_predict [ [v], [w]]
# pred_control_val = np.array([[   1,   1,   1],
#                              [   0,   0,   0]])

# control_val_R = np.zeros( len(pred_control_val[0])*2)
# state_val_Q = np.identity( len(pred_control_val[0])*3)

# state_val_Q = np.array([1,1,0.05,2,2,0.05,3,3,0.1])
# state_val_Q = np.diag(state_val_Q)

# obstacle_coords = np.array([ [x1],
#                              [y1] ])




# ---------------------- state equations ----------------------------------------
#
#                       k+1 step
#                      .
#                   .
#                .
#             .
#          .
#       .
#     K step - (X_k, Y_k, theta_k) 
#  ( v_k, w_k)

# x_k+1 = x_k + v_k.cos(theta_k).dt
# y_k+1 = y_k + v_k.sin(theta_k).dt
# theta_k+1 = theta_k + w_k.dt

# | X_k+1     |   | 1  0  0 | | X_k     |     | cos(theta_k).dt   0 | | V_k |
# | Y_k+1     | = | 0  1  0 |.| Y_k     |  +  | sin(theta_k).dt   0 |.| w_k |
# | theta_k+1 |   | 0  0  1 | | theta_k |     |               0  dt | 


# ----------------------- consider a static object ---------------------------------
#
#                    static object - ( X_s, Y_s)
#                    ..
#                  . .
#                .  .
#              .   .
#            .    .
#     l_k  .     . l_k+1
#        .      . 
#      .        k+1 step - (X_k+1, Y_k+1, theta_k+1)             
#    . 
# K step - (X_k, Y_k, theta_k)      
#   (v_k, w_k)


# orientation between,     K+1 step & K step  =  theta_k
# angle to static object,       alpha_k       =  aniSin[ ly_k/l_k]
# ori & static object angle,                  =  alphs_k - theta_k
#                                     l_k+1   =  l_k  -  v_k. cos( alpha_k - theta_k) . dt    ; consider small dts
#
#                                     
# x_k+1     = x_k + v_k.cos(theta_k).dt
# y_k+1     = y_k + v_k.sin(theta_k).dt
# theta_k+1 = theta_k + w_k.dt
# l_k+1     = l_k  -  v_k.cos( alpha_k - theta_k).dt
# ly_k+1    = ly_k - v_k.sin(theta_k).dt

# | X_k+1     |   | 1  0  0  0  0 | | X_k     |     |            cos(theta_k).dt   0 | | V_k |
# | Y_k+1     | = | 0  1  0  0  0 |.| Y_k     |  +  |            sin(theta_k).dt   0 |.| w_k |
# | theta_k+1 |   | 0  0  1  0  0 | | theta_k |     |                          0  dt | 
# | l_k+1     | = | 0  0  0  1  0 | | l_k     |     |-cos( alpha_k - theta_k).dt   0 |
# | ly_k+1    | = | 0  0  0  0  1 | | ly_k    |     |           -sin(theta_k).dt   0 |

# Ak in non linear equation
def AkFn( state_val, control_val, dt) :
    x_k     = state_val[0][0]
    y_k     = state_val[1][0]
    theta_k = state_val[2][0]
    l_k     = state_val[3][0]
    ly_k    = state_val[4][0]

    v_k     = control_val[0][0]
    w_k     = control_val[1][0]

    return np.array( [ [1, 0, 0, 0, 0], 
                       [0, 1, 0, 0, 0], 
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 1] ])

# Bk in non linear equation
def BkFn( state_val, control_val, dt) :
    x_k     = state_val[0][0]
    y_k     = state_val[1][0]
    theta_k = state_val[2][0]
    l_k     = state_val[3][0]
    ly_k    = state_val[4][0]

    v_k     = control_val[0][0]
    w_k     = control_val[1][0]

    if (l_k == 0):
        alpha_k = math.asin(1)
    elif ( (ly_k/l_k) <= -1) :
        alpha_k = math.asin(-1)
    elif ( (ly_k/ l_k) >= 1) :
        alpha_k = math.asin(1)
    else:
        alpha_k = math.asin(ly_k/l_k)
    

    return np.array( [ [         math.cos(theta_k)*dt, 0 ],
                       [         math.sin(theta_k)*dt, 0 ],
                       [                            0, dt],
                       [-math.cos(alpha_k-theta_k)*dt, 0 ],
                       [        -math.sin(theta_k)*dt, 0 ] ])

# state value obtain from nonlinear state equation
def Ak_Bk_constant( init_state, pred_control_val, dt):

    state_val = init_state
    dt = dt
    pred_control_val = pred_control_val
    Ak = np.empty([0,5,5])
    Bk = np.empty([0,5,2])
    pred_state_val = init_state

    for i in range( len(pred_control_val[0])) :
        control_val = np.array([ [pred_control_val[0][i]], [pred_control_val[1][i]]])
        A_i = AkFn( state_val, control_val, dt)
        B_i = BkFn( state_val, control_val, dt)
        X_ipluse1 = np.dot(A_i , state_val) + np.dot(B_i , control_val)
        state_val = X_ipluse1

        Ak = np.vstack( (Ak, A_i[np.newaxis, :, :]))
        Bk = np.vstack( (Bk, B_i[np.newaxis, :, :]))
        pred_state_val = np.hstack( (pred_state_val, X_ipluse1))

    # print("Ak_Bk_constant ")
    # print( pred_state_val.round(decimals=3))
    # print(Ak)
    # print(Bk)

    return Ak, Bk, pred_state_val




# ----------------- linearize state equations ------------------------------------
# x_k+1     = x_k     +  v_k.cos(theta_k).dt  -  v_k.sin(theta_k).dt.theta_k
# y_k+1     = y_k     +  v_k.sin(theta_k).dt  +  v_k.cos(theta_k).dt.theta_k
# theta_k+1 = theta_k +  w_k.dt
# l_k+1     = l_k     -  v_k.cos(alpha_k - theta_k).dt  - v_k.sin(alpha_k - theta_k).dt.theta_k - [v_k.sin(alpha_k-theta_k).dt].l_k/[cos(alpha_k).l_k.l_k] + [v_k.sin(alpha_k-theta_k).dt].ly_k/cos(alpha_k)
# ly_k+1    = ly_k    -  v_k.sin(theta_k).dt  -  v_k.cos(theta_k).dt.theta_k


# | X_k+1     |   | 1  0           -v_k.sin(theta_k).dt                                                            0                                                0 | | X_k     |     |            cos(theta_k).dt   0 | | V_k |
# | Y_k+1     | = | 0  1            v_k.cos(theta_k).dt                                                            0                                                0 |.| Y_k     |  +  |            sin(theta_k).dt   0 |.| w_k |
# | theta_k+1 |   | 0  0                              1                                                            0                                                0 | | theta_k |     |                          0  dt | 
# | l_k+1     | = | 0  0  - v_k.sin(alpha_k-theta_k).dt  1-[v_k.sin(alpha_k-theta_k).ly_k.dt]/[cos(alpha_k).l_k.l_k]  [v_k.sin(alpha_k-theta_k).dt]/[cos(alpha_k).ly_k] | | l_k   |     |-cos( alpha_k - theta_k).dt   0 |
# | ly_k+1    | = | 0  0           -v_k.cos(theta_k).dt                                                            0                                                1 | | ly_k    |     |           -sin(theta_k).dt   0 |

#     X_1    =           A_0                 . X_0         +          B_0            . U_0
#     X_2    =           A_1                 . X_1         +          B_1            . U_1

# Ak in linear equation
def Ak_linearFn( state_val, control_val, dt) :
    x_k     = state_val[0][0]
    y_k     = state_val[1][0]
    theta_k = state_val[2][0]
    l_k     = state_val[3][0]
    ly_k    = state_val[4][0]

    v_k     = control_val[0][0]
    w_k     = control_val[1][0]

    if (l_k == 0):
        alpha_k = math.asin(1)
    elif ( (ly_k/l_k) <= -1) :
        alpha_k = math.asin(-1)
    elif ( (ly_k/ l_k) >= 1) :
        alpha_k = math.asin(1)
    else:
        alpha_k = math.asin(ly_k/l_k)

    return np.array( [ [1, 0,         -(v_k*math.sin(theta_k)*dt),                                                                     0,                                                          0], 
                       [0, 1,          (v_k*math.cos(theta_k)*dt),                                                                     0,                                                          0], 
                       [0, 0,                                   1,                                                                     0,                                                          0],
                       [0, 0, -(v_k*math.sin(alpha_k-theta_k)*dt), 1-(v_k*math.sin(alpha_k-theta_k)*ly_k*dt)/(math.cos(alpha_k)*l_k*l_k), (v_k*math.sin(alpha_k-theta_k)*dt)/(math.cos(alpha_k)*l_k)],
                       [0, 0,         -(v_k*math.cos(theta_k)*dt),                                                                     0,                                                           1] ])

# Bk in linear equation
def Bk_linearFn( state_val, control_val, dt) :
    x_k     = state_val[0][0]
    y_k     = state_val[1][0]
    theta_k = state_val[2][0]
    l_k     = state_val[3][0]
    ly_k    = state_val[4][0]

    v_k     = control_val[0][0]
    w_k     = control_val[1][0]

    if (l_k == 0):
        alpha_k = math.asin(1)
    elif ( (ly_k/l_k) <= -1) :
        alpha_k = math.asin(-1)
    elif ( (ly_k/ l_k) >= 1) :
        alpha_k = math.asin(1)
    else:
        alpha_k = math.asin(ly_k/l_k)

    return np.array( [ [         math.cos(theta_k)*dt, 0 ],
                       [         math.sin(theta_k)*dt, 0 ],
                       [                            0, dt],
                       [-math.cos(alpha_k-theta_k)*dt, 0 ],
                       [        -math.sin(theta_k)*dt, 0 ] ])

 
# obtain A, B values related to nonlinear equation
def Ak_Bk_linConst( init_state, pred_control_val, dt):

    state_val = init_state
    dt = dt
    pred_control_val = pred_control_val
    Ak_linear = np.empty([0,5,5])
    Bk_linear = np.empty([0,5,2])
    pred_state_val_linear = init_state

    Ak, Bk, pred_state_val = Ak_Bk_constant(init_state, pred_control_val, dt)


    for i in range( len(pred_control_val[0])):
        control_val = np.array([ [pred_control_val[0][i]], [pred_control_val[1][i]]])
        state_val = np.array([ [pred_state_val[0][i+1]], [pred_state_val[1][i+1]], [pred_state_val[2][i+1]], [pred_state_val[3][i+1]], [pred_state_val[4][i+1]]])

        A_i_linConst = Ak_linearFn( state_val, control_val, dt)
        B_i_linConst = Bk_linearFn( state_val, control_val, dt)
        X_ipluse1 = np.dot(A_i_linConst , state_val) + np.dot(B_i_linConst , control_val)

        Ak_linear = np.vstack( (Ak_linear, A_i_linConst[np.newaxis, :, :]))
        Bk_linear = np.vstack( (Bk_linear, B_i_linConst[np.newaxis, :, :]))
        pred_state_val_linear = np.hstack( (pred_state_val_linear, X_ipluse1))

    # print("Ak_Bk_linearConst")
    # print(pred_state_val_linear.round(decimals=3))
    # print(Ak_linear)
    # print(Bk_linear)

    return Ak_linear, Bk_linear, pred_state_val_linear




# ----------------------- STATE MATRIX ------------------------------------------
# X_predict = phi . x_0 + tau .U_predict  

# X_predict = [ x_1, y_1, theta_1, l_1, ly_1, x_2, y_2, theta_2, l_2, ly_2 x_3, y_3, theta_3, l_3, ly_3 x_4, y_4, theta_4 l_4, ly_4] ^ T

# phi = [          A0]
#       [       A1.A0]
#       [    A2.A1.A0]
#       [ A3.A2.A1.A0]

# tau = [          B0,         0,       0,      0]
#       [       A1.B0,        B1,       0,      0]    
#       [    A2.A1.B0,     A2.B1,      B2,      0]  
#       [ A3.A2.A1.B0,  A3.A2.B1,   A3.B2,     B3]

# obtain phi and tau values related to linear equation
def phi_tau_constant( num_state, init_state, pred_control_val, dt):
    Ak_linear, Bk_linear, pred_state_val_linear = Ak_Bk_linConst( init_state, pred_control_val, dt)

    # phi design
    phi = np.empty([0,num_state])
    base_matrix = np.identity(num_state)
    for Ak_linears in Ak_linear:
        base_matrix = np.dot( Ak_linears, base_matrix)
        phi = np.vstack( (phi, base_matrix))
    print("phi", phi.round(decimals=2))

    # Tau design
    tau = np.empty([0, len(Ak_linear)*2])
    for i in range( len(Ak_linear)):
        # B value raw design
        # 1-[ B0,  0,  0,  0], 2-[ B0, B1,  0,  0], 3-[ B0, B1, B2,  0], 3-[ B0, B1, B2, B3]
        # print("\nB raw loop ", i)
        tau_Braw = np.empty([0,num_state,2])
        for j in range(i+1):
            tau_Braw = np.vstack( (tau_Braw, Bk_linear[j][np.newaxis, :, :]))
        for j in range(len(Ak_linear)-i-1):
            tau_Braw = np.vstack( (tau_Braw, np.zeros([num_state,2])[np.newaxis, :, :]))
        # print(tau_Braw.round(decimals=2))

        # A value raw design
        # 1-[ I,  I,  I,  I], 2-[ A1, I,  I,  I], 3-[ A2.A1, A2, I,  I], 3-[ A3.A2.A1, A3.A2, A3, I]
        # print("\nA raw loop ", i)
        tau_Araw = np.empty([0,num_state,num_state])
        for j in range(i,0,-1):
            Aks = np.identity(num_state)
            for c in range(j):
                Aks = np.dot(Aks, Ak_linear[i-c])
            tau_Araw = np.vstack( (tau_Araw, Aks[np.newaxis, :, :]))
        for j in range(len(Ak_linear)-i):
            tau_Araw = np.vstack( (tau_Araw, np.identity(num_state)[np.newaxis, :, :]))
        # print(tau_Araw.round(decimals=2))


        # multiply raw elements
        # 1-[ B0, 0, 0, 0], 2-[ A1.B0, B1, 0, 0], 3-[ A2.A1.B0, A2.B1, B2, 0], 3-[ A3.A2.A1.B0, A3.A2.B1, A3.B2, B3]
        tau_raw = np.empty([num_state,0])
        for j in range(len(tau_Araw)):
            tau_raw = np.hstack( (tau_raw, np.dot(tau_Araw[j], tau_Braw[j])))
        # print(tau_raw.round(decimals=2))


        tau = np.vstack( (tau, tau_raw))

    print("tau", tau.round(decimals=2))

    return phi, tau




# --------------------------- Cost Fn --------------------------------------
# J = (X_predict - X_ref)^T . Q . (X_predict - X_ref) + U_predict^T . R . U_predict
#   = (1/2) . U_predict^T . H . U_predict + f^T . U_predict + constant

# H = tau^T . Q . tau + R
# f = tau^T . Q . ( phi . x_0 - X_ref)

# obtain QP matrix value H and F
def QP_H_F_constant( num_state, init_state, pred_control_val, ref_state_val, control_val_R, state_val_Q, dt):

    phi, tau = phi_tau_constant( num_state, init_state, pred_control_val, dt)
    ref_state_val = ref_state_val.flatten(order='F').reshape( (len(ref_state_val[0])*num_state), 1)

    QP_H = np.dot( tau.T , state_val_Q)
    QP_H = np.dot( QP_H, tau)
    QP_H = QP_H + control_val_R

    QP_F = np.dot( phi, init_state) - ref_state_val[num_state: len(pred_control_val[0])*num_state + num_state,:]
    QP_F = np.dot( state_val_Q , QP_F) 
    QP_F = np.dot( tau.T , QP_F) 


    # print("QP_H", QP_H.round(decimals=2))
    # print("QP_F", QP_F.round(decimals=2))

    return QP_H, QP_F





# -------------------------- daqp general method ------------------------------
# (xstar,fval,exitflag,info) = daqp.solve(H,f,A,bupper,blower,sense)


# Solving the QP problem
def QP_solutions( num_state, init_state, pred_control_val, ref_state_val, control_val_R, state_val_Q, dt):
    H, F = QP_H_F_constant( num_state, init_state, pred_control_val, ref_state_val, control_val_R, state_val_Q, dt)

    H = np.array( H, dtype=c_double)
    F = np.array( F.flatten(), dtype=c_double )

    A = np.identity( len(pred_control_val[0])*2)
    A = np.array(A, dtype=c_double)

    bupper = np.tile( [3,2], len(pred_control_val[0]))
    bupper = np.array( bupper, dtype=c_double)

    blower = np.tile( [-0.1,-2], len(pred_control_val[0]))
    blower= np.array( blower, dtype=c_double)

    sense = np.zeros(6)
    sense = np.array( sense, dtype=c_int)

    x,fval,exitflag,info = daqp.solve(H,F,A,bupper,blower,sense)
    # print("Optimal solution:")
    # print(x)
    print("Exit flag:",exitflag)
    print("Info:",info)
    control_val = x.reshape( len(pred_control_val[0]),2).T
    print("control_val", control_val.round(decimals=3))
    Ak, Bk, state_value = Ak_Bk_constant( init_state, control_val, dt)
    print("state_value", state_value.round(decimals=3))

    return control_val, state_value




# init_state = np.array( [ [0], [0], [0], [5], [4]])
# pred_control_val = np.array( [ [1,1,1],[0,0,0]])
# dt = 0.1
# control_val_R = np.identity( len(pred_control_val[0])*2) *0.05

# ref_state_val = np.array([[ 0,  0.1,  0.2,  0.3],
#                           [ 0,    0,    0,    0],
#                           [ 0,    0,    0,    0],
#                           [ 4,    4,    4,    4],
#                           [ 3,    3,    3,    3], ])

# state_val_Q = np.array([ [    1,   5,   10],
#                          [    1,   5,   10],
#                          [ 0.05, 0.1, 0.15],
#                          [    0,   0,    0],
#                          [    0,   0,    0]])
# state_val_Q = state_val_Q.flatten(order='F')
# state_val_Q = np.diag(state_val_Q)

# QP_solutions( 5, init_state, pred_control_val, ref_state_val, control_val_R, state_val_Q, dt)


