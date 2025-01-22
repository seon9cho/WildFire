import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la

def bfgs(Df, x0, maxiter=100, tol=1e-5):
    """Use BFGS to minimize a function f:R^n -> R.

    Parameters:
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        maxiter (int): The maximum number of iterations to compute.
        tol (float): The stopping tolerance.

    Returns:
        ((n,) ndarray): The approximate optimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    # Keep track of convergence and number of iterations
    converged = False
    iterations = 0
    n = len(x0)
    A_inv = np.eye(n)
    dfx = Df(x0)
    # Loop maxiter number of times before convergence 
    for i in range(maxiter):
        iterations += 1
        # Equation 12.6
        x1 = x0 - A_inv@dfx
        # Condition for convergence
        if la.norm(dfx) < tol:
            converged = True
            break
        # Store all necessary values for efficiency
        s = x1 - x0
        dfx1 = Df(x1)
        y = dfx1 - dfx
        dfx = dfx1
        sy = np.dot(s,y)
        # Avoid dividing by 0
        if sy == 0:
            break
        # Equation 12.7
        A_inv += ((np.dot(s,y) + np.dot(y, A_inv@y))*np.outer(s,s))/(sy**2) \
                 - (A_inv@np.outer(y,s) + np.outer(s,y)@A_inv)/sy
        # Update x0
        x0 = x1

    return x1, converged, iterations

def f(U1, U0, K1, K2, dx, ha_i, ca_i, da_i, hb_i, cb_i, db_i):
    """
    The nonlinear implicit Crank-Nicolson equations for 
    the Burgers' equation.
    
    Parameters
    ----------
        U1 (ndarray): The values of U^(n+1)
        U0 (ndarray): The values of U^n
        s (float): wave speed
        K1 (float): first constant in the equations
        K2 (float): second constant in the equations
    
    Returns
    ----------
        out (ndarray): The residuals (differences between right- and 
                    left-hand sides) of the equation, accounting 
                    for boundary conditions
    """
    # Coefficients for the boundary conditions
    coef1_a = ha_i / (ca_i - da_i/dx)
    coef2_a = da_i / (dx*ca_i - da_i)
    coef1_b = hb_i / (cb_i + db_i/dx)
    coef2_b = db_i / (dx*cb_i + db_i)
    # Implicit Crank-Nicolson scheme
    RHS = U1[1:-1] - U0[1:-1]
    LHS1 = K1*((U1[1:-1] + U0[1:-1]) * \
               (U1[2:] - U1[:-2] + U0[2:] - U0[:-2]))
    LHS2 = K2*(U1[2:] - 2*U1[1:-1] + U1[:-2] + \
               U0[2:] - 2*U0[1:-1] + U0[:-2])
    # Solve for the values at the interior and the boundary points
    residual = RHS - (LHS1 + LHS2)
    boundary_a = U1[0] - (coef1_a - coef2_a*U1[1])
    boundary_b = U1[-1] - (coef1_b - coef2_b*U1[-2])
    # Return the interior with the two boundary values
    return np.concatenate([[boundary_a], residual, [boundary_b]])

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

def burgers_equation(a,b,T,N_x,N_t,u_0,c_a,d_a,h_a,c_b,d_b,h_b):
    """
    Solve the Burgers equation of the following form using the Crank-Nicolson scheme:
    u_t + u*u_x = u_xx, x ∈ [a, b], t ∈ (0, T]
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
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    # Compute the given functions at each point in time
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
    # Compute the coefficients necessary for the implicit Crank-Nicolson scheme
    K1 = -dt / (8*dx)
    K2 = dt / (2*dx**2)
    # Find the boundary values when t=0 using u_0
    U0 = u_0(x)
    # List to contain the points at each timestep
    U = [U0]
    for i in range(N_t - 1):
        # Define a new function g that only takes in U1 using f
        def g(U1):
            return f(U1, U0, K1, K2, dx, ha[i], ca[i], da[i], hb[i], cb[i], db[i])
        # Solve for U1 using BFGS
        U1, conv, iterations = bfgs(g, U0)
        U.append(U1)
        U0 = U1
    
    return np.array(U)

def test():
    # Traveling wave problem as the test case
    a = -20
    b = 20
    T = 1.0
    N_x = 150
    N_t = 350
    u_hat = lambda x: 3 - 2*np.tanh(2*x/2)
    v = lambda x: 3.5*(np.sin(3*x) + 1) * (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)
    u_0 = lambda x: u_hat(x) + v(x)
    c_a = lambda t: 1
    d_a = lambda t: 0
    h_a = lambda t: 5
    c_b = lambda t: 1
    d_b = lambda t: 0
    h_b = lambda t: 1
    U_burgers = burgers_equation(a,b,T,N_x,N_t,u_0,c_a,d_a,h_a,c_b,d_b,h_b)
    # Animate the result
    x = np.linspace(a, b, N_x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim((a, b))
    ax.set_ylim((0, 6))
    traj, = plt.plot([],[], color='r')
    particle, = plt.plot([],[], marker='o', color='r')
    def update(i):
        traj.set_data(x, U_burgers[i])
        return particle, traj

    ani = animation.FuncAnimation(fig, update, frames=range(N_t), interval=10)
    plt.show()

