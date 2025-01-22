import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D
import unittest

def check_error(a,b,T,N_x,N_t,u_0,c_a,d_a,h_a,c_b,d_b,h_b):
    """
    Function that handles exceptions for the heat_equation function.
    Raises necessary error when the input parameters of heat_equation are not correct.
    """
 
    from types import FunctionType

    if type(a) != float and type(a) != int: raise TypeError("a must be a float or an int")
    if type(b) != float and type(b) != int: raise TypeError("b must be a float or an int")
    if type(T) != float and type(T) != int: raise TypeError("T must be a float or an int")

    if type(N_x) != int: raise TypeError("N_x must be an int")
    if type(N_t) != int: raise TypeError("N_t must be an int")

    if type(u_0) != FunctionType: raise TypeError("u_0 must be a function")
    if type(c_a) != FunctionType: raise TypeError("c_a must be a function")
    if type(d_a) != FunctionType: raise TypeError("d_a must be a function")
    if type(h_a) != FunctionType: raise TypeError("h_a must be a function")
    if type(c_b) != FunctionType: raise TypeError("c_b must be a function")
    if type(d_b) != FunctionType: raise TypeError("d_b must be a function")
    if type(h_b) != FunctionType: raise TypeError("h_b must be a function")

    if a >= b: raise ValueError("b must be strictly greater than a")
    if T <= 0: raise ValueError("T must be positive")
    if N_x <= 2: raise ValueError("N_x must be greater than 1")
    if N_t <= 1: raise ValueError("N_t must be greater than 0")

def heat_equation(a,b,T,N_x,N_t,u_0,c_a,d_a,h_a,c_b,d_b,h_b):
    """
    Solve the heat equation of the following form using the Crank-Nicolson scheme:
    u_t = u_xx, x ∈ [a, b], t ∈ (0, T]
    u(x, 0) = u_0(x),
    h_a(t) = c_a(t)u(a, t) + d_a(t)u_x(a, t)
    h_b(t) = c_b(t)u(b, t) + d_b(t)u_x(b, t).
    
    Parameters:
    a: float
    b: float, a < b
    T: positive float
    N_x: positive integer, N_x > 2, N_x = numer of mesh nodes in x
    N_t: positive integer, N_t > 1, N_t = number of mesh nodes in t
    u_0: function handle for the initial function auxiliary condition
    c_a: function handle
    d_a: function handle
    h_a: function handle
    c_b: function handle
    d_b: function handle
    h_b: function handle
    
    Returns:
    U: a two dimensional numpy array containing floats.
       Rows correspond to time and columns to x.
    """
    # Check for errors in the input parameter
    check_error(a,b,T,N_x,N_t,u_0,c_a,d_a,h_a,c_b,d_b,h_b)
    # Create points for spatial and time domain
    x = np.linspace(a, b, N_x)
    t = np.linspace(0, T, N_t)
    # Calculate lambda
    h = x[1] - x[0]
    k = t[1] - t[0]
    lmda = k / (2*h**2)
    # Compute the give functions at each point in time
    ha = h_a(t)
    ca = c_a(t)
    da = d_a(t)
    hb = h_b(t)
    cb = c_b(t)
    db = d_b(t)
    # If the function handle received returns a constant output, convert it to be
    # a vector of those constants
    if type(ha) == int or type(ha) == float: ha = ha*np.ones(N_t)
    if type(ca) == int or type(ca) == float: ca = ca*np.ones(N_t)
    if type(da) == int or type(da) == float: da = da*np.ones(N_t)
    if type(hb) == int or type(hb) == float: hb = hb*np.ones(N_t)
    if type(cb) == int or type(cb) == float: cb = cb*np.ones(N_t)
    if type(db) == int or type(db) == float: db = db*np.ones(N_t)
    # Initialize A and B sparse diagonal matrix that iteratively solves the heat
    # equation using the Crank-Nicolson scheme
    A = sp.diags([lmda, 1-2*lmda, lmda], [-1, 0, 1], shape=(N_x-2, N_x-2))
    B = sp.diags([-lmda, 1+2*lmda, -lmda], [-1, 0, 1], shape=(N_x-2, N_x-2))
    # Find the boundary values when t=0 using u_0
    U_j = u_0(x)
    # List to contain the points at every timesteps
    U = [U_j]
    for i in range(N_t - 1):
        # Boundary values of the previous series
        U_at = U_j[0]
        U_bt = U_j[-1]
        # Interior values of the previous series
        U_ = U_j[1:-1]
        # Coefficients for the boundary condition
        coef1_a = ha[i] / (ca[i] - da[i]/h)
        coef2_a = da[i] / (h*ca[i] - da[i])
        coef1_b = hb[i] / (cb[i] + db[i]/h)
        coef2_b = db[i] / (h*cb[i] + db[i])
        # Apply the boundary condition to the Crank-Nicolson matrices
        AU = A@U_
        AU[0] += lmda*U_at + lmda*coef1_a
        AU[-1] += lmda*U_bt + lmda*coef1_b
        
        B_ = B.tocsr().copy()
        B_[0,0] -= lmda*coef2_a
        B_[-1,-1] -= lmda*coef2_b 
        # Solve the linear system for the interior values
        U1_ = spsolve(B_, AU)
        # Solve for the boundary values
        Ua = coef1_a - coef2_a * U1_[0]
        Ub = coef1_b - coef2_b * U1_[-1]
        # Concatenate then append
        U_j = np.concatenate([[Ua], U1_, [Ub]])
        U.append(U_j)
        
    return np.array(U)

def test():
    a = 0
    b = np.pi
    T = 1.0
    N_x = 30
    N_t = 10
    u_0 = lambda x: np.sin(x)
    c_a = lambda t:1
    d_a = lambda t:0
    h_a = lambda t:0
    c_b = lambda t:1
    d_b = lambda t:0
    h_b = lambda t:0
    U = heat_equation(a,b,T,N_x,N_t,u_0,c_a,d_a,h_a,c_b,d_b,h_b)
    x = np.linspace(a, b, N_x)
    t = np.linspace(0, T, N_t)
    X,t = np.meshgrid(x, t)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, t, U, cmap="coolwarm")
    plt.show()







