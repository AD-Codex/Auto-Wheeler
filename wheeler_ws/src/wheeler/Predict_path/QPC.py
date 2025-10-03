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
def AkFn( num_state, state_val, control_val, dt) :
    x_k     = state_val[0][0]
    y_k     = state_val[1][0]
    theta_k = state_val[2][0]

    AkFn = np.array([1,1,1], dtype=np.float64)

    for i in range( (num_state - 3)//2 ):
        AkFn = np.append( AkFn, [1,1])

    v_k     = control_val[0][0]
    w_k     = control_val[1][0]

    return np.diag(AkFn)

# Bk in non linear equation
def BkFn( num_state, state_val, control_val, dt) :
    x_k     = state_val[0][0]
    y_k     = state_val[1][0]
    theta_k = state_val[2][0]

    v_k     = control_val[0][0]
    w_k     = control_val[1][0]

    BkFn    = np.array( [ [ math.cos(theta_k)*dt, 0 ],
                          [ math.sin(theta_k)*dt, 0 ],
                          [                    0, dt] ], dtype=np.float64)
    
    objs = (num_state - 3)//2
    for i in range( objs):
        l_k  = state_val[ 2*(i+ 1) + 1 ][0]
        ly_k = state_val[ 2*(i+ 1) + 2 ][0]

        if (l_k == 0):
            alpha_k = math.asin(1)
        elif ( (ly_k/l_k) <= -1) :
            alpha_k = math.asin(-1)
        elif ( (ly_k/ l_k) >= 1) :
            alpha_k = math.asin(1)
        else:
            alpha_k = math.asin(ly_k/l_k)

        BkFn = np.vstack( ( BkFn, np.array([ -math.cos(alpha_k-theta_k)*dt, 0], dtype=np.float64)))
        BkFn = np.vstack( ( BkFn, np.array([         -math.sin(theta_k)*dt, 0], dtype=np.float64)))
    
    return BkFn

# state value obtain from nonlinear state equation
def Ak_Bk_constant( num_state, init_state, pred_control_val, dt):
    state_val = init_state
    Ak = np.empty([ 0, num_state, num_state], dtype=np.float64)
    Bk = np.empty([ 0, num_state,         2], dtype=np.float64)
    pred_state_val = init_state

    for i in range( len(pred_control_val[0])) :
        control_val = np.array([ [pred_control_val[0][i]], [pred_control_val[1][i]]])
        A_i = AkFn( num_state, state_val, control_val, dt)
        B_i = BkFn( num_state, state_val, control_val, dt)
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
def Ak_linearFn( num_state, state_val, control_val, dt) :
    x_k     = state_val[0][0]
    y_k     = state_val[1][0]
    theta_k = state_val[2][0]

    v_k     = control_val[0][0]
    w_k     = control_val[1][0]

    Ak_linear_val = np.array([1,1,1], dtype=np.float64)
    for i in range( (num_state - 3)//2 ):
        Ak_linear_val = np.append( Ak_linear_val, [1,1])
    Ak_linear_val = np.diag(Ak_linear_val)

    Ak_linear_val[0][2] = -(v_k*math.sin(theta_k)*dt)
    Ak_linear_val[1][2] =  (v_k*math.cos(theta_k)*dt)

    objs = (num_state - 3)//2
    for i in range( objs):
        l_k  = state_val[ 2*(i+ 1) + 1 ][0]
        ly_k = state_val[ 2*(i+ 1) + 2 ][0]

        if (l_k == 0):
            alpha_k = math.asin(1)
        elif ( (ly_k/l_k) <= -1) :
            alpha_k = math.asin(-1)
        elif ( (ly_k/ l_k) >= 1) :
            alpha_k = math.asin(1)
        else:
            alpha_k = math.asin(ly_k/l_k)

        Ak_linear_val[ 2*(i+ 1) + 1][            2] =                                   -(v_k*math.sin(alpha_k-theta_k)*dt)
        Ak_linear_val[ 2*(i+ 1) + 1][ 2*(i+ 1) + 1] = 1-(v_k*math.sin(alpha_k-theta_k)*ly_k*dt)/(math.cos(alpha_k)*l_k*l_k)
        Ak_linear_val[ 2*(i+ 1) + 1][ 2*(i+ 1) + 2] =            (v_k*math.sin(alpha_k-theta_k)*dt)/(math.cos(alpha_k)*l_k)
        Ak_linear_val[ 2*(i+ 1) + 2][            2] =                                           -(v_k*math.cos(theta_k)*dt)

    return Ak_linear_val

# Bk in linear equation
def Bk_linearFn( num_state, state_val, control_val, dt) :
    x_k     = state_val[0][0]
    y_k     = state_val[1][0]
    theta_k = state_val[2][0]

    v_k     = control_val[0][0]
    w_k     = control_val[1][0]

    Bk_linearFn    = np.array( [ [ math.cos(theta_k)*dt, 0 ],
                                 [ math.sin(theta_k)*dt, 0 ],
                                 [                    0, dt] ], dtype=np.float64)
    
    objs = (num_state - 3)//2
    for i in range( objs):
        l_k  = state_val[ 2*(i+ 1) + 1 ][0]
        ly_k = state_val[ 2*(i+ 1) + 2 ][0]

        if (l_k == 0):
            alpha_k = math.asin(1)
        elif ( (ly_k/l_k) <= -1) :
            alpha_k = math.asin(-1)
        elif ( (ly_k/ l_k) >= 1) :
            alpha_k = math.asin(1)
        else:
            alpha_k = math.asin(ly_k/l_k)

        Bk_linearFn = np.vstack( ( Bk_linearFn, np.array([ -math.cos(alpha_k-theta_k)*dt, 0]) ))
        Bk_linearFn = np.vstack( ( Bk_linearFn, np.array([         -math.sin(theta_k)*dt, 0]) ))

    return Bk_linearFn

# obtain A, B values related to nonlinear equation
def Ak_Bk_linConst( num_state, init_state, pred_control_val, dt):
    state_val = init_state
    dt = dt
    pred_control_val = pred_control_val
    Ak_linear = np.empty([ 0, num_state, num_state], dtype=np.float64)
    Bk_linear = np.empty([ 0, num_state,         2], dtype=np.float64)
    pred_state_val_linear = init_state

    Ak, Bk, cal_state_val = Ak_Bk_constant( num_state, init_state, pred_control_val, dt)
    # print(cal_state_val)

    for i in range( len(pred_control_val[0])):
        control_val = pred_control_val[:,i][:, np.newaxis]
        state_val = cal_state_val[:,i][:, np.newaxis]

        A_i_linConst = Ak_linearFn( num_state, state_val, control_val, dt)
        B_i_linConst = Bk_linearFn( num_state, state_val, control_val, dt)
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
    Ak_linear, Bk_linear, pred_state_val_linear = Ak_Bk_linConst( num_state, init_state, pred_control_val, dt)

    # phi design
    phi = np.empty([0,num_state], dtype=np.float64)
    base_matrix = np.identity(num_state)
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
def QP_solutions( num_state, init_state, pred_control_val, ref_state_val, control_val_R, state_val_Q, dt):
    H, F = QP_H_F_constant( num_state, init_state, pred_control_val, ref_state_val, control_val_R, state_val_Q, dt)

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
    min_cost = 0.5 * x.T @ H @ x + F.T @ x
    print("Minimum cost:", min_cost)
    control_val = x.reshape( len(pred_control_val[0]),2).T
    print("control_val", control_val.round(decimals=3))
    Ak, Bk, state_value = Ak_Bk_constant( num_state, init_state, control_val, dt)
    print("state_value", state_value.round(decimals=3))

    return control_val, state_value





# -------------------------- daqp contour method -----------------------------
# Solving the QP problem with contuor controlling
def QPC_solutions( num_state, init_state, pred_control_val, ref_state_val, control_val_R, state_val_Q, dt):
    Q_matrix = contour_constant( num_state, pred_control_val, ref_state_val, state_val_Q)

    # print(Q_matrix)
    H, F = QP_H_F_constant( num_state, init_state, pred_control_val, ref_state_val, control_val_R, Q_matrix, dt)

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
    bupper = np.tile([3.0, 6.0], N)
    blower = np.tile([-0.1, -6.0], N)

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
    Ak, Bk, state_value = Ak_Bk_constant( num_state, init_state, control_val, dt)
    print("state_value", state_value.round(decimals=3))

    return control_val, state_value





# X_0 = np.array([ [0], [0], [-0.05], [0.5], [0.05], [0.8], [0.05]], dtype=np.float64)
# dt = 0.1
# pred_control_val = np.array([[   1,   1,   1],
#                              [   0,   0,   0]], dtype=np.float64)

# ref_state_val = np.array([[0, 1, 1.1, 1.2], 
#                           [0, 0,   0,   0],
#                           [0, 0,   0,   0],
#                           [0, 0,   0,   0],
#                           [0, 0,   0,   0],
#                           [0, 0,   0,   0], 
#                           [0, 0,   0,   0]])

# control_val_R = np.zeros( len(pred_control_val[0])*2)

# state_val_Q = np.array([1,1,0.05,0,0,0,0,2,2,0.05,0,0,0,0,3,3,0.1,0,0,0,0])
# state_val_Q = np.diag(state_val_Q)

# QP_solutions( 7, X_0, pred_control_val, ref_state_val, control_val_R, state_val_Q, dt)




