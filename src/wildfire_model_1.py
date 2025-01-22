import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la
from tqdm import tqdm
from scipy.optimize import fsolve
from scipy.optimize import root

def f(TS1, TS0, K1, K2, K3, K4, B, C1, dx, N_x,
      hTa_i, cTa_i, dTa_i, hTb_i, cTb_i, dTb_i, 
      hSa_i, cSa_i, dSa_i, hSb_i, cSb_i, dSb_i):
    """
    The nonlinear implicit Crank-Nicholson equations for 
    the Wildfire model.
    
    Parameters
    ----------
        TS1 (ndarray): stacked array of the value for T^n+1 and S^n+1
        TS0 (ndarray): stacked array of the value for T^n and S^n
        K1 (float): first constant 
        K2 (float): second constant
        K3 (float): third constant
        K4 (float): fourth constant
        B: float
        C1: flaot
        dx: delta x
        N_x: number of space meshes
        hTa_i: boundary value of hT_a at i
        cTa_i: boundary value of cT_a at i
        dTa_i: boundary value of dT_a at i
        hTb_i: boundary value of hT_b at i
        cTb_i: boundary value of cT_b at i
        dTb_i: boundary value of dT_b at i
        hSa_i: boundary value of hS_a at i
        cSa_i: boundary value of cS_a at i
        dSa_i: boundary value of dS_a at i
        hSb_i: boundary value of hS_b at i
        cSb_i: boundary value of cS_b at i
        dSb_i: boundary value of dS_b at i     


    Returns
    ----------
        out (ndarray): The residuals (differences between right- and 
                    left-hand sides) of the equation, accounting 
                    for boundary conditions
    """
    T0 = TS0[:N_x]
    S0 = TS0[N_x:]
    T1 = TS1[:N_x]
    S1 = TS1[N_x:]

    RHS_T = T1[1:-1] - T0[1:-1]
    LHS1_T = K1*(T1[:-2] - 2*T1[1:-1] + T1[2:] +
                 T0[:-2] - 2*T0[1:-1] + T0[2:])
    LHS2_T = K2*(T1[2:] - T1[:-2] + T0[2:] - T0[:-2])
    LHS3_T = K3*((S1[1:-1] + S0[1:-1]) * 
                np.exp(-2*B / (T1[1:-1] + T0[1:-1]))
                - C1*(T1[1:-1] + T0[1:-1]))
    residual_T = RHS_T - (LHS1_T + LHS2_T + LHS3_T)
    
    RHS_S = S1[1:-1] - S0[1:-1]
    LHS_S = K4*(S1[1:-1] + S0[1:-1]) * \
               np.exp(-2*B / (T1[1:-1] + T0[1:-1]))
    residual_S = RHS_S - LHS_S

    #boundary_Ta = hTa_i - cTa_i*T1[0] - dTa_i*(T1[1] - T1[0]) / dx
    #boundary_Tb = hTb_i - cTb_i*T1[-1] - dTb_i*(T1[-1] - T1[-2]) / dx
    #boundary_Sa = hSa_i - cSa_i*S1[0] - dSa_i*(S1[1] - S1[0]) / dx
    #boundary_Sb = hSb_i - cSb_i*S1[-1] - dSb_i*(S1[-1] - S1[-2]) / dx

    boundary_Ta = hTa_i - cTa_i*(T1[0]+T0[0])/2 - \
                dTa_i*(T1[1] - T1[0] + T0[1] - T0[0]) / (2*dx)
    boundary_Tb = hTb_i - cTb_i*(T1[-1]+T0[-1])/2 - \
                dTb_i*(T1[-1] - T1[-2] + T0[-1] - T0[-2]) / (2*dx)
    boundary_Sa = hSa_i - cSa_i*(S1[0]+S0[0])/2 - \
                dSa_i*(S1[1] - S1[0] + S0[1] - S0[0]) / (2*dx)
    boundary_Sb = hSb_i - cSb_i*(S1[-1]+S0[-1])/2 - \
                dSb_i*(S1[-1] - S1[-2] + S0[-1] - S0[-2]) / (2*dx)
    
    T_out = np.hstack([boundary_Ta, residual_T, boundary_Tb])
    S_out = np.hstack([boundary_Sa, residual_S, boundary_Sb])
    return np.hstack([T_out, S_out])

def check_error(a,b,T,N_x,N_t,T_0,S_0,cT_a,dT_a,hT_a,cT_b,dT_b,hT_b,
                cS_a,dS_a,hS_a,cS_b,dS_b,hS_b,A,B,C1,C2,nu):
    """
    Function that handles exceptions for the heat_equation function.
    Raises necessary error when the input parameters of heat_equation are not correct.
    """
    from types import FunctionType

    if type(a) != float and type(a) != int: raise TypeError("a must be a float or an int")
    if type(b) != float and type(b) != int: raise TypeError("b must be a float or an int")
    if type(T) != float and type(T) != int: raise TypeError("T must be a float or an int")
    if type(A) != float and type(A) != int: raise TypeError("A must be a float or an int")
    if type(B) != float and type(B) != int: raise TypeError("B must be a float or an int")
    if type(C1) != float and type(C1) != int: raise TypeError("C1 must be a float or an int")
    if type(C2) != float and type(C2) != int: raise TypeError("C2 must be a float or an int")
    if type(nu) != float and type(nu) != int: raise TypeError("nu must be a float or an int")

    if type(N_x) != int: raise TypeError("N_x must be an int")
    if type(N_t) != int: raise TypeError("N_t must be an int")

    if type(T_0) != FunctionType: raise TypeError("T_0 must be a function")
    if type(S_0) != FunctionType: raise TypeError("S_0 must be a function")
    if type(cT_a) != FunctionType: raise TypeError("cT_a must be a function")
    if type(dT_a) != FunctionType: raise TypeError("dT_a must be a function")
    if type(hT_a) != FunctionType: raise TypeError("hT_a must be a function")
    if type(cT_b) != FunctionType: raise TypeError("cT_b must be a function")
    if type(dT_b) != FunctionType: raise TypeError("dT_b must be a function")
    if type(hT_b) != FunctionType: raise TypeError("hT_b must be a function")
    if type(cS_a) != FunctionType: raise TypeError("cS_a must be a function")
    if type(dS_a) != FunctionType: raise TypeError("dS_a must be a function")
    if type(hS_a) != FunctionType: raise TypeError("hS_a must be a function")
    if type(cS_b) != FunctionType: raise TypeError("cS_b must be a function")
    if type(dS_b) != FunctionType: raise TypeError("dS_b must be a function")
    if type(hS_b) != FunctionType: raise TypeError("hS_b must be a function")

    if a >= b: raise ValueError("b must be strictly greater than a")
    if T <= 0: raise ValueError("T must be positive")
    if N_x <= 2: raise ValueError("N_x must be greater than 1")
    if N_t <= 1: raise ValueError("N_t must be greater than 0")

def wildfire_model(a,b,T,N_x,N_t,T_0,S_0,cT_a,dT_a,hT_a,cT_b,dT_b,hT_b,
                   cS_a,dS_a,hS_a,cS_b,dS_b,hS_b,A,B,C1,C2,nu):
    """
    Solve the Wildfire model of the following form using the Crank-Nicolson scheme:
    T_t = T_xx - ν*T_x + A*(S*exp(−B/T) − C1*T),
    S_t = -C2*S*exp(-B/T), x ∈ [a, b], t ∈ (0, T]
    T(x, 0) = T_0(x),
    S(x, 0) = S_0(x),
    hT_a(t) = cT_a(t)T(a, t) + dT_a(t)T_x(a, t),
    hT_b(t) = cT_b(t)T(b, t) + dT_b(t)T_x(b, t),
    hS_a(t) = cS_a(t)S(a, t) + dS_a(t)S_x(a, t),
    hS_b(t) = cS_b(t)S(b, t) + dS_b(t)S_x(b, t).
    
    Parameters:
    a: float
    b: float, a < b
    T: positive float
    N_x: positive integer, N_x > 2, N_x = number of mesh nodes in x
    N_t: positive integer, N_t > 1, N_t = number of mesh nodes in t
    T_0: function handle for the initial function auxiliary condition for T
    S_0: function handle for the initial function auxiliary condition for S
    cT_a: function handle
    dT_a: function handle
    hT_a: function handle
    cT_b: function handle
    dT_b: function handle
    hT_b: function handle
    cS_a: function handle
    dS_a: function handle
    hS_a: function handle
    cS_b: function handle
    dS_b: function handle
    hS_b: function handle
    A: float
    B: float
    C1: float
    C2: float
    nu: float (ν)
    
    Returns
    T: a two dimensional numpy array containing floats.
       Rows correspond to time and columns to x for T.
    S: a two dimensional numpy array containing floats.
       Rows correspond to time and columns to x for S.
    
    """
    # Check for errors in the input parameter
    check_error(a,b,T,N_x,N_t,T_0,S_0,cT_a,dT_a,hT_a,cT_b,dT_b,hT_b,
                cS_a,dS_a,hS_a,cS_b,dS_b,hS_b,A,B,C1,C2,nu)
    # Create points for spatial and time domain
    x, dx = np.linspace(a, b, N_x, retstep=True)
    t, dt = np.linspace(0, T, N_t, retstep=True)
    # Compute the given functions at each point in time
    hTa = hT_a(t)
    cTa = cT_a(t)
    dTa = dT_a(t)
    hTb = hT_b(t)
    cTb = cT_b(t)
    dTb = dT_b(t)
    
    hSa = hS_a(t)
    cSa = cS_a(t)
    dSa = dS_a(t)
    hSb = hS_b(t)
    cSb = cS_b(t)
    dSb = dS_b(t)
    # If the function handle received returns a constant output, convert it to be
    # a vector of those constants
    if type(hTa) == int or type(hTa) == float: hTa = hTa*np.ones(N_t)
    if type(cTa) == int or type(cTa) == float: cTa = cTa*np.ones(N_t)
    if type(dTa) == int or type(dTa) == float: dTa = dTa*np.ones(N_t)
    if type(hTb) == int or type(hTb) == float: hTb = hTb*np.ones(N_t)
    if type(cTb) == int or type(cTb) == float: cTb = cTb*np.ones(N_t)
    if type(dTb) == int or type(dTb) == float: dTb = dTb*np.ones(N_t)

    if type(hSa) == int or type(hSa) == float: hSa = hSa*np.ones(N_t)
    if type(cSa) == int or type(cSa) == float: cSa = cSa*np.ones(N_t)
    if type(dSa) == int or type(dSa) == float: dSa = dSa*np.ones(N_t)
    if type(hSb) == int or type(hSb) == float: hSb = hSb*np.ones(N_t)
    if type(cSb) == int or type(cSb) == float: cSb = cSb*np.ones(N_t)
    if type(dSb) == int or type(dSb) == float: dSb = dSb*np.ones(N_t)
    # Compute the coefficients necessary for the implicit Crank-Nicolson scheme
    K1 = dt / (2*dx**2)
    K2 = -nu*dt / (4*dx)
    K3 = A*dt/2
    K4 = -C2*dt/2
    # Find the boundary values when t=0 using T_0 and S_0
    T0 = T_0(x)
    T = [T0]
    S0 = S_0(x)
    S = [S0]
    loop = tqdm(total=N_t, position=0, leave=False)
    for i in range(N_t - 1):
        # Define a new function g that only takes in TS1 using f
        def _f(TS1):
            return f(TS1, TS0, K1, K2, K3, K4, B, C1, dx, N_x,
                     hTa[i], cTa[i], dTa[i], hTb[i], cTb[i], dTb[i], 
                     hSa[i], cSa[i], dSa[i], hSb[i], cSb[i], dSb[i])
        
        TS0 = np.hstack([T0, S0])
        # Solve for TS1 using fsolve
        TS1 = fsolve(_f, TS0)
        # Update T0 and S0 and append to list
        T0 = TS1[:N_x]
        S0 = TS1[N_x:]
        T.append(T0)
        S.append(S0)
        loop.update(1)
    loop.close()
    return np.array(T), np.array(S)