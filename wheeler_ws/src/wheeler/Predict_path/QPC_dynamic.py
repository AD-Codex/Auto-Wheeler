# customizable mpc controller with object avoiding
# ros2 launch foxglove_bridge foxglove_bridge_launch.xml


import daqp
from ctypes import * 
import ctypes.util
import numpy as np
import math




# X_0 = np.array([ [1], [2], [0]])
# dt = 0.1
# dt should be very small

# # reference state values [[ x],[ y],[ theta], [l], [l_k], [l], [l_k]]
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


# ----------------------- consider a dynanic object ---------------------------------
# object move toward to robot
#
#                                (Xs_k, Ys_k)
#         (Xs_k+1, Ys_k+1)      .
#                       .    .
#                      .  . 
#                     . .
#                   ..
#                .  .
#     l_k     .    . l_k+1
#          .      . 
#       .           k+1 step - (X_k+1, Y_k+1, theta_k+1)             
#    .     
# K step - (X_k, Y_k, theta_k)      
#
# robot movement    ( v_k, w_k)
# object movement   ( X_dot_k, Y_dot_k), positive if move toward robot

# robot to object distance,     l_(k+1)_x   =   l_k_x - X_dot_k.dt - v_k.cos(theta_k).dt
#                               l_(k+1)_y   =   l_k_y - Y_dot_k.dt - v_k.sin(theta_k).dt

#                                     
# x_k+1     = x_k + v_k.cos(theta_k).dt
# y_k+1     = y_k + v_k.sin(theta_k).dt
# theta_k+1 = theta_k + w_k.dt
# l_(k+1)_x = l_k_x - v_k.cos(theta_k).dt - X_dot_k.dt
# l_(k+1)_y = l_k_y - v_k.sin(theta_k).dt - Y_dot_k.dt 

# | X_k+1     |   | 1  0  0  0  0 | | X_k     |     |  cos(theta_k).dt   0 | | V_k |     |   0    0 | | X_dot_k |
# | Y_k+1     | = | 0  1  0  0  0 |.| Y_k     |  +  |  sin(theta_k).dt   0 |.| w_k |  +  |   0    0 |.| Y_dot_k |
# | theta_k+1 |   | 0  0  1  0  0 | | theta_k |     |                0  dt |             |   0    0 |
# | l_(k+1)_x | = | 0  0  0  1  0 | | l_k_x   |     | -cos(theta_k).dt   0 |             | -dt    0 |
# | l_(k+1)_y | = | 0  0  0  0  1 | | l_k_y   |     | -sin(theta_k).dt   0 |             |   0  -dt |
 
#     X_1       =             A_0  .  X_0       +                      B_0  .  U_0    +         D_0  .  W_0
#     X_2       =             A_1  .  X_1       +                      B_1  .  U_1    +         D_1  .  W_1

# Ak in non linear equation
def AkFn( num_state, state_val, control_val, dt) :
    AkFn = np.diag( np.ones( num_state, dtype=np.float64))

    return AkFn

# Bk in non linear equation
def BkFn( num_state, state_val, control_val, dt) :
    theta_k = state_val[2][0]

    BkFn    = np.array( [ [ math.cos(theta_k)*dt, 0 ],
                          [ math.sin(theta_k)*dt, 0 ],
                          [                    0, dt] ], dtype=np.float64)
    
    objs = (num_state - 3)//2
    for i in range( objs):
        BkFn_update = np.array([[ -math.cos(theta_k)*dt, 0],
                                [ -math.sin(theta_k)*dt, 0]], dtype=np.float64)

        BkFn = np.vstack( ( BkFn, BkFn_update))
    
    return BkFn

# Dk in non linear equation
def DkFn( num_state, state_val, control_val, dt):
    objs = (num_state - 3)//2

    DkFn = np.zeros((3, 2*objs), dtype=np.float64)
    DkFn_update = np.diag([-dt]*2*objs) 
    DkFn = np.vstack( (DkFn, DkFn_update))

    return DkFn

# state value obtain from nonlinear state equation
def Ak_Bk_Dk_constant( num_state, pred_horizon, init_state, pred_control_val, obj_vel, dt):
    objs = (num_state - 3)//2
    Ak = np.empty([ 0, num_state, num_state], dtype=np.float64)
    Bk = np.empty([ 0, num_state,         2], dtype=np.float64)
    Dk = np.empty([ 0, num_state,    2*objs], dtype=np.float64)

    state_val_i     = init_state
    pred_state_val  = state_val_i
    for i in range( pred_horizon) :
        control_val_i   = pred_control_val[:,i][:, np.newaxis]

        A_i = AkFn( num_state, state_val_i, control_val_i, dt)
        B_i = BkFn( num_state, state_val_i, control_val_i, dt)
        D_i = DkFn( num_state, state_val_i, control_val_i, dt)

        if ( D_i.size == 0):
            state_val_i_plus_1 = np.dot( A_i, state_val_i) + np.dot( B_i, control_val_i)
        else :
            obj_vel_i       = obj_vel[:,i][:, np.newaxis]
            state_val_i_plus_1 = np.dot( A_i, state_val_i) + np.dot( B_i, control_val_i) + np.dot(D_i, obj_vel_i)

        Ak = np.vstack( (Ak, A_i[np.newaxis, :, :]))
        Bk = np.vstack( (Bk, B_i[np.newaxis, :, :]))
        Dk = np.vstack( (Dk, D_i[np.newaxis, :, :]))
        pred_state_val = np.hstack( (pred_state_val, state_val_i_plus_1))

        state_val_i = state_val_i_plus_1

    # print("Ak_Bk_constant ")
    # print( pred_state_val.round(decimals=3))
    # print(Ak)
    # print(Bk)
    # print(Dk)

    return Ak, Bk, Dk, pred_state_val





# ----------------- linearize state equations ------------------------------------
# x_k+1     = x_k     +  v_k.cos(theta_k).dt  -  v_k.sin(theta_k).dt.theta_k
# y_k+1     = y_k     +  v_k.sin(theta_k).dt  +  v_k.cos(theta_k).dt.theta_k
# theta_k+1 = theta_k +  w_k.dt
# l_(k+1)_x = l_k_x   -  v_k.cos(theta_k).dt  +  v_k.sin(theta_k).dt.theta_k - X_dot_k.dt
# l_(k+1)_y = l_k_y   -  v_k.sin(theta_k).dt  -  v_k.cos(theta_k).dt.theta_k - Y_dot_k.dt 


# | X_k+1     |   | 1  0  -v_k.sin(theta_k).dt  0   0 | | X_k     |     |  cos(theta_k).dt   0 | | V_k |     |   0    0 | | X_dot_k |
# | Y_k+1     | = | 0  1   v_k.cos(theta_k).dt  0   0 |.| Y_k     |  +  |  sin(theta_k).dt   0 |.| w_k |  +  |   0    0 |.| Y_dot_k |
# | theta_k+1 |   | 0  0                     1  0   0 | | theta_k |     |                0  dt |             |   0    0 |
# | l_(k+1)_x | = | 0  0   v_k.sin(theta_k).dt  1   0 | | l_k_x   |     | -cos(theta_k).dt   0 |             | -dt    0 |
# | l_(k+1)_y | = | 0  0  -v_k.cos(theta_k).dt  0   1 | | l_k_y   |     | -sin(theta_k).dt   0 |             |   0  -dt |


#     X_1       =                                 A_0  .  X_0       +                      B_0  .  U_0    +         D_0  .  W_0
#     X_2       =                                 A_1  .  X_1       +                      B_1  .  U_1    +         D_1  .  W_1



# Ak in linear equation
def Ak_linearFn( num_state, state_val, control_val, dt) :
    theta_k = state_val[2][0]
    v_k     = control_val[0][0]

    Ak_linear_val = np.diag( np.ones( num_state, dtype=np.float64))

    Ak_linear_val[0][2] = -(v_k*math.sin(theta_k)*dt)
    Ak_linear_val[1][2] =  (v_k*math.cos(theta_k)*dt)

    objs = (num_state - 3)//2
    for i in range( objs):
        Ak_linear_val[ 3+2*i][2] =  (v_k*math.sin(theta_k)*dt)
        Ak_linear_val[ 4+2*i][2] = -(v_k*math.cos(theta_k)*dt)

    return Ak_linear_val

# Bk in linear equation
def Bk_linearFn( num_state, state_val, control_val, dt) :
    theta_k = state_val[2][0]

    Bk_linearFn    = np.array( [ [ math.cos(theta_k)*dt, 0 ],
                                 [ math.sin(theta_k)*dt, 0 ],
                                 [                    0, dt] ], dtype=np.float64)
    
    objs = (num_state - 3)//2
    for i in range( objs):
        BkFn_update = np.array([[ -math.cos(theta_k)*dt, 0],
                                [ -math.sin(theta_k)*dt, 0]], dtype=np.float64)

        Bk_linearFn = np.vstack( ( Bk_linearFn, BkFn_update))

    return Bk_linearFn

# Dk in linear equation
def Dk_linearFn( num_state, state_val, control_val, dt):
    objs = (num_state - 3)//2

    Dk_linearFn = np.zeros(( 3, 2*objs), dtype=np.float64)
    DkFn_update = np.diag( [-dt]*2*objs) 
    Dk_linearFn = np.vstack( (Dk_linearFn, DkFn_update))

    return Dk_linearFn

# obtain A, B, D values related to nonlinear equation
def Ak_Bk_Dk_linConst( num_state, pred_horizon, init_state, pred_control_val, obj_vel, dt):
    objs = (num_state - 3)//2
    Ak_linear = np.empty([ 0, num_state, num_state], dtype=np.float64)
    Bk_linear = np.empty([ 0, num_state,         2], dtype=np.float64)
    Dk_linear = np.empty([ 0, num_state,    2*objs], dtype=np.float64)

    Ak, Bk, Dk, cal_state_val = Ak_Bk_Dk_constant( num_state, pred_horizon, init_state, pred_control_val, obj_vel, dt)

    pred_state_val_linear = init_state
    for i in range( pred_horizon):
        control_val_i   = pred_control_val[:,i][:, np.newaxis]
        state_val_i     = cal_state_val[:,i][:, np.newaxis]

        A_i_linConst = Ak_linearFn( num_state, state_val_i, control_val_i, dt)
        B_i_linConst = Bk_linearFn( num_state, state_val_i, control_val_i, dt)
        D_i_linConst = Dk_linearFn( num_state, state_val_i, control_val_i, dt)

        if ( D_i_linConst.size == 0):
            X_ipluse1 = np.dot( A_i_linConst, state_val_i) + np.dot( B_i_linConst, control_val_i)
        else :
            obj_vel_i       = obj_vel[:,i][:, np.newaxis]
            X_ipluse1 = np.dot( A_i_linConst, state_val_i) + np.dot( B_i_linConst, control_val_i) + np.dot( D_i_linConst, obj_vel_i)

        Ak_linear = np.vstack( (Ak_linear, A_i_linConst[np.newaxis, :, :]))
        Bk_linear = np.vstack( (Bk_linear, B_i_linConst[np.newaxis, :, :]))
        Dk_linear = np.vstack( (Dk_linear, D_i_linConst[np.newaxis, :, :]))

        pred_state_val_linear = np.hstack( (pred_state_val_linear, X_ipluse1))

    # print("Ak_Bk_linearConst")
    # print(pred_state_val_linear.round(decimals=3))
    # print(Ak_linear)
    # print(Bk_linear)
    # print(Dk_linear)

    return Ak_linear, Bk_linear, Dk_linear, pred_state_val_linear




# ----------------------- STATE MATRIX ------------------------------------------
# X_predict = phi . x_0 + tau .U_predict +  eta .W_value

# X_predict = [ x_1, y_1, theta_1, l_1_x, l_1_y, x_2, y_2, theta_2, l_2_x, l_2_y, x_3, y_3, theta_3, l_3_x, l_3_y, x_4, y_4, theta_4, l_4_x, l_4_y] ^ T

# phi = [          A0]
#       [       A1.A0]
#       [    A2.A1.A0]
#       [ A3.A2.A1.A0]

# tau = [          B0,         0,       0,      0]
#       [       A1.B0,        B1,       0,      0]    
#       [    A2.A1.B0,     A2.B1,      B2,      0]  
#       [ A3.A2.A1.B0,  A3.A2.B1,   A3.B2,     B3]

# eta = [          D0,         0,       0,      0]
#       [       A1.D0,        D1,       0,      0]    
#       [    A2.A1.D0,     A2.D1,      D2,      0]  
#       [ A3.A2.A1.D0,  A3.A2.D1,   A3.D2,     D3]

# obtain phi and tau values related to linear equation
def phi_tau_constant( num_state, pred_horizon, init_state, pred_control_val, obj_vel, dt):
    Ak_linear, Bk_linear, Dk_linear, pred_state_val_linear = Ak_Bk_Dk_linConst( num_state, pred_horizon, init_state, pred_control_val, obj_vel, dt)

    # phi design
    phi = np.empty( [ 0,num_state], dtype=np.float64)
    base_matrix = np.identity( num_state)
    for Ak_linears in Ak_linear:
        base_matrix = np.dot( Ak_linears, base_matrix)
        phi = np.vstack( (phi, base_matrix))
    # print("phi", phi.round(decimals=2))

    # Tau design
    tau = np.empty([0, len(Ak_linear)*2], dtype=np.float64)
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
    # print("tau", tau.round(decimals=2))

    # Eta design
    objs = (num_state - 3)//2
    eta = np.empty([0, len(Ak_linear)*2*objs], dtype=np.float64)
    for i in range( len(Ak_linear)):
        # D value raw design
        # 1-[ D0,  0,  0,  0], 2-[ D0, D1,  0,  0], 3-[ D0, D1, D2,  0], 3-[ D0, D1, D2, D3]
        # print("\nD raw loop ", i)
        eta_Draw = np.empty([ 0, num_state, 2*objs])
        for j in range(i+1):
            eta_Draw = np.vstack( (eta_Draw, Dk_linear[j][np.newaxis, :, :]))
        for j in range(len(Ak_linear)-i-1):
            eta_Draw = np.vstack( (eta_Draw, np.zeros([ num_state, 2*objs])[np.newaxis, :, :]))
        # print(eta_Draw.round(decimals=4))

        # A value raw design
        # 1-[ I,  I,  I,  I], 2-[ A1, I,  I,  I], 3-[ A2.A1, A2, I,  I], 3-[ A3.A2.A1, A3.A2, A3, I]
        # print("\nA raw loop ", i)
        eta_Araw = np.empty([0,num_state,num_state])
        for j in range(i,0,-1):
            Aks = np.identity(num_state)
            for c in range(j):
                Aks = np.dot(Aks, Ak_linear[i-c])
            eta_Araw = np.vstack( (eta_Araw, Aks[np.newaxis, :, :]))
        for j in range(len(Ak_linear)-i):
            eta_Araw = np.vstack( (eta_Araw, np.identity(num_state)[np.newaxis, :, :]))
        # print(eta_Araw.round(decimals=2))


        # multiply raw elements
        # 1-[ B0, 0, 0, 0], 2-[ A1.B0, B1, 0, 0], 3-[ A2.A1.B0, A2.B1, B2, 0], 3-[ A3.A2.A1.B0, A3.A2.B1, A3.B2, B3]
        eta_raw = np.empty([num_state,0])
        for j in range(len(eta_Araw)):
            eta_raw = np.hstack( (eta_raw, np.dot(eta_Araw[j], eta_Draw[j])))
        # print(eta_raw.round(decimals=2))


        eta = np.vstack( (eta, eta_raw))
    # print("eta", eta.round(decimals=2))

    return phi, tau, eta



# --------------------------- Cost Fn --------------------------------------
# J = (X_predict - X_ref)^T . Q . (X_predict - X_ref) + U_predict^T . R . U_predict
#   = (1/2) . U_predict^T . H . U_predict + f^T . U_predict + constant

# H = tau^T . Q . tau + R
# f = tau^T . Q . ( phi . x_0 + eta . W_value - X_ref)

# obtain QP matrix value H and F
def QP_H_F_constant( num_state, pred_horizon, init_state, pred_control_val, obj_vel, ref_state_val, control_val_R, state_val_Q, dt):

    phi, tau, eta = phi_tau_constant( num_state, pred_horizon, init_state, pred_control_val, obj_vel, dt)
    ref_state_val = ref_state_val.flatten(order='F').reshape( ref_state_val.size, 1)
    obj_vel       = obj_vel.flatten(order='F').reshape( obj_vel.size, 1)

    QP_H = np.dot( tau.T , state_val_Q)
    QP_H = np.dot( QP_H, tau)
    QP_H = QP_H + control_val_R

    QP_F = np.dot( phi, init_state) + np.dot(eta, obj_vel) - ref_state_val[num_state: len(pred_control_val[0])*num_state + num_state,:]
    QP_F = np.dot( state_val_Q , QP_F) 
    QP_F = np.dot( tau.T , QP_F) 


    # print("QP_H", QP_H.round(decimals=2))
    # print("QP_F", QP_F.round(decimals=2))

    return QP_H, QP_F




# ------------------    contour control --------------------------------
# error_1 - parallel shift
# error_2 - distance shift
# error_3 - orientation error

# error_1 = [ -sin(theta_r)   cos(theta_r)   0].[         x_1 - x_ref ]
#                                               [         y_1 - y_ref ]
#                                               [ theta_1 - theta_ref ]

# error_2 = [  cos(theta_r)   sin(theta_r)   0].[         x_1 - x_ref ]
#                                               [         y_1 - y_ref ]
#                                               [ theta_1 - theta_ref ]

# error_3 = [  0   0   1].[         x_1 - x_ref ]
#                         [         y_1 - y_ref ]
#                         [ theta_1 - theta_ref ]

# | error_1 |    | -sin(theta_r)   cos(theta_r)   0 |   |         x_1 - x_ref |
# | error_2 | =  |  cos(theta_r)   sin(theta_r)   0 | . |         y_1 - y_ref |
# | error_3 |    |             0              0   1 |   | theta_1 - theta_ref |
 
#  [ Error ]  =                   S                   .    [ X_pred - X_ref]

#  [ Error ]^T . Q . [ Error ] = [ X_pred - X_ref]^T  .  S^T  .  Q  .  S  .  [ X_pred - X_ref]

# J = [ Error ]^T . Q . [ Error ] + U_predict^T . R . U_predict
# J = [ X_pred - X_ref]^T  .  Q_  .  [ X_pred - X_ref] + U_predict^T . R . U_predict
# Q_ = S^T  .  Q  .  S 

# Q matrix obtain
def contour_constant( num_state, pred_control_val, ref_state_val, state_val_Q) :
    num_oca = len(pred_control_val[0])
    Q_matrix = np.zeros((num_oca*num_state, num_oca*num_state))
    for i in range( num_oca):
        ref_theta = ref_state_val[2][i+1]
        S = np.diag( np.ones(num_state))
        S[0:2, 0:2] = np.array([[ -np.sin( ref_theta), np.cos( ref_theta)],
                                [  np.cos( ref_theta), np.sin( ref_theta)]])
        
        Q_matrix[ i*num_state: i*num_state+num_state, i*num_state: i*num_state+num_state] = S
        # print(S)

    Q_ = np.dot( Q_matrix.T , state_val_Q)
    Q_ = np.dot( Q_ , Q_matrix)

    return Q_





# -------------------------- daqp general method ------------------------------
# (xstar,fval,exitflag,info) = daqp.solve(H,f,A,bupper,blower,sense)


# Solving the QP problem
def QP_solutions( num_state, pred_horizon, init_state, pred_control_val, obj_vel, ref_state_val, control_val_R, state_val_Q, dt):
    H, F = QP_H_F_constant( num_state, pred_horizon, init_state, pred_control_val, obj_vel, ref_state_val, control_val_R, state_val_Q, dt)

    # Number of control inputs per time step (e.g., v, w → 2), and prediction horizon steps
    N = pred_control_val.shape[1]
    n_vars = 2 * N       # total variables: [v0, w0, v1, w1, ..., vN-1, wN-1]
    n_cons = 2 * N       # constraints: upper/lower for v and w at each step

    # all inputs are correctly typed and memory-aligned
    H = np.ascontiguousarray(H, dtype=np.float64)
    F = np.ascontiguousarray(F.flatten(), dtype=np.float64)

    # Constraint matrix A (identity for box constraints)
    A = np.ascontiguousarray(np.eye(n_cons), dtype=np.float64)

    # Upper and lower bounds for each control input: [v <= 3, w <= 1]
    bupper = np.tile([3.0, 5.0], N)
    blower = np.tile([-0.1, -5.0], N)

    bupper = np.ascontiguousarray(bupper, dtype=np.float64)
    blower = np.ascontiguousarray(blower, dtype=np.float64)

    # Constraint types: 0 for inequality (Ax ≤ b) — same length as number of constraints
    sense = np.ascontiguousarray(np.zeros(n_cons, dtype=np.int32))

    x,fval,exitflag,info = daqp.solve(H,F,A,bupper,blower,sense)
    # print("Optimal solution:")
    # print(x)
    print("Exit flag:",exitflag)
    print("Info:",info)
    control_val = x.reshape( len(pred_control_val[0]),2).T
    print("control_val", control_val.round(decimals=3))
    Ak, Bk, Dk, state_val = Ak_Bk_Dk_constant( num_state, pred_horizon, init_state, control_val, obj_vel, dt)
    print("state_value", state_val.round(decimals=3))

    return control_val, state_val


# -------------------------- daqp contour method -----------------------------
# Solving the QP problem with contuor controlling
def QPC_solutions( num_state, pred_horizon, init_state, pred_control_val, obj_vel, ref_state_val, control_val_R, state_val_Q, dt):
    Q_matrix = contour_constant( num_state, pred_control_val, ref_state_val, state_val_Q)

    # print(Q_matrix)
    H, F = QP_H_F_constant( num_state, pred_horizon, init_state, pred_control_val, obj_vel, ref_state_val, control_val_R, Q_matrix, dt)

    # Number of control inputs per time step (e.g., v, w → 2), and prediction horizon steps
    N = pred_control_val.shape[1]
    n_vars = 2 * N       # total variables: [v0, w0, v1, w1, ..., vN-1, wN-1]
    n_cons = 2 * N       # constraints: upper/lower for v and w at each step

    # all inputs are correctly typed and memory-aligned
    H = np.ascontiguousarray(H, dtype=np.float64)
    F = np.ascontiguousarray(F.flatten(), dtype=np.float64)

    # Constraint matrix A (identity for box constraints)
    A = np.ascontiguousarray(np.eye(n_cons), dtype=np.float64)

    # Upper and lower bounds for each control input: [v <= 3, w <= 1]
    bupper = np.tile([3.0, 5.0], N)
    blower = np.tile([-0.1, -5.0], N)

    bupper = np.ascontiguousarray(bupper, dtype=np.float64)
    blower = np.ascontiguousarray(blower, dtype=np.float64)

    # Constraint types: 0 for inequality (Ax ≤ b) — same length as number of constraints
    sense = np.ascontiguousarray(np.zeros(n_cons, dtype=np.int32))

    x,fval,exitflag,info = daqp.solve(H,F,A,bupper,blower,sense)
    print("Optimal solution:")
    print("Info:",info)
    print("Exit flag:",exitflag)
    min_cost = 0.5 * x.T @ H @ x + F.T @ x
    print("Minimum cost:", min_cost)
    control_val = x.reshape( len(pred_control_val[0]),2).T
    print("control_val", control_val.round(decimals=3))
    Ak, Bk, Dk, state_val = Ak_Bk_Dk_constant( num_state, pred_horizon, init_state, control_val, obj_vel, dt)
    print("state_value", state_val.round(decimals=3))

    return control_val, state_val



# num_state = 7
# pred_horizon = 3
# init_state = np.array([ [0], [0], [0.1], [1], [0.01], [-1], [0.01]] , dtype=np.float64)
# dt = 0.025
# pred_control_val = np.array([[   1,   1,   1],
#                              [   0,   0,   0]] , dtype=np.float64)

# obj_vel = np.array( [[ 0, 0, 0],
#                      [ 0, 0, 0]] , dtype=np.float64)

# obj_vel = np.array( [[ 1, 1, 1],
#                      [ 0, 0, 0],
#                      [ 0, 0, 0],
#                      [ 1, 1, 1]] , dtype=np.float64)

# ref_state_val = np.array([[0,0.05, 0.1,0.15], 
#                           [0,   0,   0,   0], 
#                           [0,   0,   0,   0],
#                           [0,   0,   0,   0], 
#                           [0,   0,   0, 0.1],
#                           [0,   0,   0,   0], 
#                           [0,   0,   0,   0]])

# control_val_R = np.zeros( len(pred_control_val[0])*2)
# state_val_Q = np.array([1,1,0.05,0,0,0,0,2,2,0.05,0,0,0,0,3,3,0.1,0,100,0,0])
# state_val_Q = np.diag(state_val_Q)

# QPC_solutions( num_state, pred_horizon, init_state, pred_control_val, obj_vel, ref_state_val, control_val_R, state_val_Q, dt)





