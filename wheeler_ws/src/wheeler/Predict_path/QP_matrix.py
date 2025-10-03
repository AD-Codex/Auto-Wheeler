import daqp
from ctypes import * 
import ctypes.util
import numpy as np
import math



# X_0 = np.array([ [1], [2], [0]])
# dt = 0.1
# dt should be very small

# # reference state values [[ x],[ y],[ z]]
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






# ---------------------- state equations ----------------------------------------
# x_k+1 = x_k + v_k.cos(theta_k).dt
# y_k+1 = y_k + v_k.sin(theta_k).dt
# theta_k+1 = theta_k + w_k.dt

# | X_k+1     |   | 1  0  0 | | X_k     |     | cos(theta_k).dt   0 | | V_k |
# | Y_k+1     | = | 0  1  0 |.| Y_k     |  +  | sin(theta_k).dt   0 |.| w_k |
# | theta_k+1 |   | 0  0  1 | | theta_k |     |               0  dt | 

# Ak in non linear equation
def AkFn( state_val, control_val, dt) :
    x_k = state_val[0][0]
    y_k = state_val[1][0]
    theta_k = state_val[2][0]
    v_k = control_val[0][0]
    w_k = control_val[1][0]

    return np.array( [ [1, 0, 0 ], 
                       [0, 1, 0 ], 
                       [0, 0, 1 ]])
# Bk in non linear equation
def BkFn( state_val, control_val, dt) :
    x_k = state_val[0][0]
    y_k = state_val[1][0]
    theta_k = state_val[2][0]
    v_k = control_val[0][0]
    w_k = control_val[1][0]

    return np.array( [ [math.cos(theta_k)*dt, 0 ],
                       [math.sin(theta_k)*dt, 0 ],
                       [                   0, dt]])

# state value obtain from nonlinear state equation
def Ak_Bk_constant( init_state, dt, pred_control_val):

    state_val = init_state
    dt = dt
    pred_control_val = pred_control_val
    Ak = np.empty([0,3,3])
    Bk = np.empty([0,3,2])
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
# x_k+1 = x_k  +  v_k.cos(theta_k).dt  -  v_k.sin(theta_k).dt.theta_k
# y_k+1 = y_k  +  v_k.sin(theta_k).dt  +  v_k.cos(theta_k).dt.theta_k
# theta_k+1 = theta_k + w_k.dt

# | X_k+1     |   | 1  0  -v_k.sin(theta_k).dt | | X_k     |     | cos(theta_k).dt   0 | | V_k |
# | Y_k+1     | = | 0  1   v_k.cos(theta_k).dt |.| Y_k     |  +  | sin(theta_k).dt   0 |.| w_k |
# | theta_k+1 |   | 0  0                     1 | | theta_k |     |               0  dt |

#     X_1    =           A_0                 . X_0         +          B_0            . U_0
#     X_2    =           A_1                 . X_1         +          B_1            . U_1

# Ak in linear equation
def Ak_linearFn( state_val, control_val, dt) :
    x_k = state_val[0][0]
    y_k = state_val[1][0]
    theta_k = state_val[2][0]
    v_k = control_val[0][0]
    w_k = control_val[1][0]

    # return np.array( [ [1, 0, 0 ], 
    #                    [0, 1, 0 ], 
    #                    [0, 0, 1 ]])

    return np.array( [ [1, 0, -(v_k*math.sin(theta_k)*dt) ], 
                       [0, 1,  (v_k*math.cos(theta_k)*dt) ], 
                       [0, 0,                           1 ]])
# Bk in linear equation
def Bk_linearFn( state_val, control_val, dt) :
    x_k = state_val[0][0]
    y_k = state_val[1][0]
    theta_k = state_val[2][0]
    v_k = control_val[0][0]
    w_k = control_val[1][0]

    return np.array( [ [math.cos(theta_k)*dt, 0 ],
                       [math.sin(theta_k)*dt, 0 ],
                       [                   0, dt]])
 
# obtain A, B values related to nonlinear equation
def Ak_Bk_linConst( init_state, dt, pred_control_val):

    state_val = init_state
    dt = dt
    pred_control_val = pred_control_val
    Ak_linear = np.empty([0,3,3])
    Bk_linear = np.empty([0,3,2])
    pred_state_val_linear = init_state

    Ak, Bk, pred_state_val = Ak_Bk_constant(init_state, dt, pred_control_val)

    for i in range( len(pred_control_val[0])):
        control_val = np.array([ [pred_control_val[0][i]], [pred_control_val[1][i]]])
        state_val = np.array([ [pred_state_val[0][i+1]], [pred_state_val[1][i+1]], [pred_state_val[2][i+1]]])

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

# X_predict = [ x_1, y_1, theta_1, x_2, y_2, theta_2, x_3, y_3, theta_3, x_4, y_4, theta_4] ^ T

# phi = [          A0]
#       [       A1.A0]
#       [    A2.A1.A0]
#       [ A3.A2.A1.A0]

# tau = [          B0,         0,       0,      0]
#       [       A1.B0,        B1,       0,      0]    
#       [    A2.A1.B0,     A2.B1,      B2,      0]  
#       [ A3.A2.A1.B0,  A3.A2.B1,   A3.B2,     B3]

# obtain phi and tau values related to linear equation
def phi_tau_constant( init_state, dt, pred_control_val):
    Ak_linear, Bk_linear, pred_state_val_linear = Ak_Bk_linConst( init_state, dt, pred_control_val)

    # phi design
    phi = np.empty([0,3])
    base_matrix = np.identity(3)
    for Ak_linears in Ak_linear:
        base_matrix = np.dot( Ak_linears, base_matrix)
        phi = np.vstack( (phi, base_matrix))
    # print("phi", phi.round(decimals=2))

    # Tau design
    tau = np.empty([0, len(Ak_linear)*2])
    for i in range( len(Ak_linear)):
        # B value raw design
        # 1-[ B0,  0,  0,  0], 2-[ B0, B1,  0,  0], 3-[ B0, B1, B2,  0], 3-[ B0, B1, B2, B3]
        # print("\nB raw loop ", i)
        tau_Braw = np.empty([0,3,2])
        for j in range(i+1):
            tau_Braw = np.vstack( (tau_Braw, Bk_linear[j][np.newaxis, :, :]))
        for j in range(len(Ak_linear)-i-1):
            tau_Braw = np.vstack( (tau_Braw, np.zeros([3,2])[np.newaxis, :, :]))
        # print(tau_Braw.round(decimals=2))

        # A value raw design
        # 1-[ I,  I,  I,  I], 2-[ A1, I,  I,  I], 3-[ A2.A1, A2, I,  I], 3-[ A3.A2.A1, A3.A2, A3, I]
        # print("\nA raw loop ", i)
        tau_Araw = np.empty([0,3,3])
        for j in range(i,0,-1):
            Aks = np.identity(3)
            for c in range(j):
                Aks = np.dot(Aks, Ak_linear[i-c])
            tau_Araw = np.vstack( (tau_Araw, Aks[np.newaxis, :, :]))
        for j in range(len(Ak_linear)-i):
            tau_Araw = np.vstack( (tau_Araw, np.identity(3)[np.newaxis, :, :]))
        # print(tau_Araw.round(decimals=2))


        # multiply raw elements
        # 1-[ B0, 0, 0, 0], 2-[ A1.B0, B1, 0, 0], 3-[ A2.A1.B0, A2.B1, B2, 0], 3-[ A3.A2.A1.B0, A3.A2.B1, A3.B2, B3]
        tau_raw = np.empty([3,0])
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
def QP_H_F_constant( init_state, dt, pred_control_val, ref_state_val, control_val_R, state_val_Q):

    phi, tau = phi_tau_constant( init_state, dt, pred_control_val)
    ref_state_val = ref_state_val.flatten(order='F').reshape( (len(ref_state_val[0])*3), 1)

    QP_H = np.dot( tau.T , state_val_Q)
    QP_H = np.dot( QP_H, tau)
    QP_H = QP_H + control_val_R

    QP_F = np.dot( phi, init_state) - ref_state_val[3: len(pred_control_val[0])*3 + 3,:]
    QP_F = np.dot( state_val_Q , QP_F) 
    QP_F = np.dot( tau.T , QP_F) 


    # print("QP_H", QP_H.round(decimals=2))
    # print("QP_F", QP_F.round(decimals=2))

    return QP_H, QP_F





# ------------------    contour control --------------------------------
# error_2 - perpendiculer distance
# error_1 - parallel distance
# error_0 - orientation error

# error_2 = [ -sin(theta_r)   cos(theta_r)   0].[         x_1 - x_ref ]
#                                               [         y_1 - y_ref ]
#                                               [ theta_1 - theta_ref ]
# error_1 = [  cos(theta_r)   sin(theta_r)   0].[         x_1 - x_ref ]
#                                               [         y_1 - y_ref ]
#                                               [ theta_1 - theta_ref ]
# error_0 = [  0   0   1].[         x_1 - x_ref ]
#                         [         y_1 - y_ref ]
#                         [ theta_1 - theta_ref ]

# | error_2 |    | -sin(theta_r)   cos(theta_r)   0 |   |         x_1 - x_ref |
# | error_1 | =  |  cos(theta_r)   sin(theta_r)   0 | . |         y_1 - y_ref |
# | error_0 |    |             0              0   1 |   | theta_1 - theta_ref |
 
#  [ Error ]  =                   S                   .    [ X_pred - X_ref]

#  [ Error ]^T . Q . [ Error ] = [ X_pred - X_ref]^T  .  S^T  .  Q  .  S  .  [ X_pred - X_ref]

# J = [ Error ]^T . Q . [ Error ] + U_predict^T . R . U_predict
# J = [ X_pred - X_ref]^T  .  Q_  .  [ X_pred - X_ref] + U_predict^T . R . U_predict
# Q_ = S^T  .  Q  .  S 

# Q matrix obtain
def contour_constant( pred_control_val, ref_state_val, state_val_Q) :
    num_oca = len(pred_control_val[0])
    Q_matrix = np.zeros((num_oca*3, num_oca*3))
    for i in range( num_oca):
        ref_theta = ref_state_val[2][i+1]
        S = np.array( [ [ -np.sin( ref_theta), np.cos( ref_theta),   0],
                        [  np.cos( ref_theta), np.sin( ref_theta),   0],
                        [                    0,                 0,   1] ])
        Q_matrix[ i*3: i*3+3, i*3: i*3+3] = S
        # print(S)

    Q_ = np.dot( Q_matrix.T , state_val_Q)
    Q_ = np.dot( Q_ , Q_matrix)

    return Q_




# -------------------------- daqp general method ------------------------------
# (xstar,fval,exitflag,info) = daqp.solve(H,f,A,bupper,blower,sense)


# Solving the QP problem
def QP_solutions(init_state, dt, pred_control_val, ref_state_val, control_val_R, state_val_Q):
    H, F = QP_H_F_constant(init_state, dt, pred_control_val, ref_state_val, control_val_R, state_val_Q)

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
    print(x)
    print("Exit flag:",exitflag)
    print("Info:",info)
    min_cost = 0.5 * x.T @ H @ x + F.T @ x
    print("Minimum cost:", min_cost)
    control_val = x.reshape( len(pred_control_val[0]),2).T
    print("control_val", control_val.round(decimals=3))
    Ak, Bk, state_value = Ak_Bk_constant( init_state, dt, control_val)
    print("state_value", state_value.round(decimals=3))

    return control_val, state_value




# -------------------------- daqp contour method -----------------------------
# Solving the QP problem with contuor controlling
def QPC_solutions(init_state, dt, pred_control_val, ref_state_val, control_val_R, state_val_Q):
    Q_matrix = contour_constant( pred_control_val, ref_state_val, state_val_Q)

    # print(Q_matrix)
    H, F = QP_H_F_constant(init_state, dt, pred_control_val, ref_state_val, control_val_R, Q_matrix)

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
    bupper = np.tile([3.0, 1.0], N)
    blower = np.tile([-0.1, -1.0], N)

    bupper = np.ascontiguousarray(bupper, dtype=np.float64)
    blower = np.ascontiguousarray(blower, dtype=np.float64)

    # Constraint types: 0 for inequality (Ax ≤ b) — same length as number of constraints
    sense = np.ascontiguousarray(np.zeros(n_cons, dtype=np.int32))

    x,fval,exitflag,info = daqp.solve(H,F,A,bupper,blower,sense)
    # print("Optimal solution:")
    # print("Info:",info)
    # print("Exit flag:",exitflag)
    min_cost = 0.5 * x.T @ H @ x + F.T @ x
    print("Minimum cost:", min_cost)
    control_val = x.reshape( len(pred_control_val[0]),2).T
    # print("control_val", control_val.round(decimals=3))
    Ak, Bk, state_value = Ak_Bk_constant( init_state, dt, control_val)
    # print("state_value", state_value.round(decimals=3))

    return control_val, state_value

