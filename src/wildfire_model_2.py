import numpy as np
from scipy.optimize import fsolve
from scipy import sparse

class mesh():
    """
    Class for handling all the details of the mesh we create.
    """
    
    def __init__(self, x1, xJ, tN, J, N):
        """
        Initialize a meshgrid object.
        x1 -- First spatial point in the grid.
        xJ -- Last spatial point in the grid.
        tN -- The last time point in the grid (first point is assumed to be 0)
        J -- The number of spatial points
        N -- The number of time points
        """
        self.x_points, self.delta_x = np.linspace(x1, xJ, J, retstep=True)
        self.t_points, self.delta_t = np.linspace(0, tN, N, retstep=True)
    
    @property
    def shape(self):
        """
        Return the number of t points and number of x points as a pair.
        """
        return (self.numberOfTPoints, self.numberOfXPoints)
    
    @property
    def numberOfXPoints(self):
        return self.x_points.size
    
    @property
    def numberOfTPoints(self):
        return self.t_points.size

class robinBoundaryCondition():
    
    def __init__(self, c, d, h):
        self.c, self.d, self.h = c, d, h
    
    def __call__(self, t):
        return self.c(t), self.d(t), self.h(t)

class wildfireSolver():
    
    def __init__(self,a,b,T,N_x,N_t,T_0,S_0,cT_a,dT_a,hT_a,cT_b,
    dT_b,hT_b,cS_a,dS_a,hS_a,cS_b,dS_b,hS_b,A,B,C1,C2,nu):
        
        self.mesh = mesh(a, b, T, N_x, N_t)
        self.boundaryConditionTA = robinBoundaryCondition(cT_a, dT_a, hT_a)
        self.boundaryConditionTB = robinBoundaryCondition(cT_b, dT_b, hT_b)
        self.boundaryConditionSA = robinBoundaryCondition(cS_a, dS_a, hS_a)
        self.boundaryConditionSB = robinBoundaryCondition(cS_b, dS_b, hS_b)
        self.initial_condition_T = T_0
        self.initial_condition_S = S_0
        self.constants = [A, B, C1, C2, nu]
        return
    
    def solve(self):
        
        T, S = self.initializeTandS()
        
        for n in range(self.mesh.numberOfTPoints - 1):
            
            print("On time step", n+2, end='\r')
        
            #Define a function that is zero when there is no error.
            rootFunc = lambda x: self.crankNicholsonError(x, n, T, S)
            
            #Interleave S and T for our initial guess ()
            T_S_inter_guess = np.empty(T.shape[1] + S.shape[1])
            T_S_inter_guess[::2] = T[n, :]
            T_S_inter_guess[1::2] = S[n, :]
            
            #Throw the initial guess and root funciton at fsolve
            T_S_interleaved = fsolve(rootFunc, T_S_inter_guess)
            
            #Un-interleave the results back into T and S.
            T[n+1, :] = T_S_interleaved[::2]
            S[n+1, :] = T_S_interleaved[1::2]
        
        return T, S
        
    def initializeTandS(self):
        S0 = np.vectorize(self.initial_condition_S)
        T0 = np.vectorize(self.initial_condition_T)
        x = self.mesh.x_points
        S = np.empty(self.mesh.shape)
        T = S.copy()
        S[0, :] = S0(x)
        T[0, :] = T0(x)
        return T, S
        
        
    def crankNicholsonError(self, x, n, T, S):
        #Unpack S and T from the vector x (they're interlaced).
        T[n+1, :] = x[::2] #T is at the even indices.
        S[n+1, :] = x[1::2] #S is t the odd indices
        
        #Unpack the constants we need from the 
        dx, dt = self.mesh.delta_x, self.mesh.delta_t
        A, B, C1, C2, nu = self.constants
        
        #Approximate all the derivatives on the interior.
        T_apx = (T[n+1, 1:-1] + T[n, 1:-1])/2
        S_apx = (S[n+1, 1:-1] + S[n, 1:-1])/2
        T_t = T[n+1, 1:-1] - T[n, 1:-1]
        S_t = S[n+1, 1:-1] - S[n, 1:-1]
        T_xx = np.mean([T[z, 2:] - 2*T[z, 1:-1] + T[z, :-2] for z in [n, n+1]],
                         axis=0)
        T_x = np.mean([T[z, 2:] - T[z, :-2] for z in (n, n+1)], axis=0)
        S_x = np.mean([S[z, 2:] - S[z, :-2] for z in (n, n+1)], axis=0)
        
        #Create the constants necessary to account for stepsize.
        K1 = nu * dt/(2*dx)
        K2 = dt/(dx**2)
        K3 = A*dt
        
        #Calculate the error on the interior.
        T_err = -T_t + K2*T_xx - K1*T_x
        T_err +=   K3 * (S_apx * np.exp(-B/T_apx) - C1*T_apx)
        S_err = -S_t - C2*dt/2 * (S_apx * np.exp(-B / T_apx)) # << don't think is correct
        
        #Calculate the error at the robin boundary condition when x=a.
        t = self.mesh.t_points[n:n+2].mean()
        c, d, h = self.boundaryConditionTA(t)
        T_x = (T[n:n+2, 1] - T[n:n+2, 0]).mean()/dx
        T_error_a = -h + c*T[n:n+2,0].mean() + d*T_x
        c, d, h = self.boundaryConditionSA(t)
        S_x = (S[n:n+2, 1] - S[n:n+2, 0]).mean()/dx
        S_error_a = -h + c*S[n:n+2,0].mean() + d*S_x
        
        #Calculate the error at the robin boundary condition when x=b
        c, d, h = self.boundaryConditionTB(t)
        T_x = (T[n:n+2,-1] - T[n:n+2,-2]).mean()/dx
        T_error_b = -h + c*T[n:n+2,-1].mean() + d*T_x
        c, d, h = self.boundaryConditionSB(t)
        S_x = (S[n:n+2,-1] - S[n:n+2,-2]).mean()/dx
        S_error_b = -h + c*S[n:n+2,-1].mean() + d*S_x
        
        #Throw on the errors from the boundary conditions.
        T_err = np.hstack((T_error_a, T_err, T_error_b))
        S_err = np.hstack((S_error_a, S_err, S_error_b))
        
        #Interleave the errors again before returning.
        interleaved = np.empty(T_err.size + S_err.size)
        interleaved[::2] = T_err #T goes at the even indices.
        interleaved[1::2] = S_err #S goes at the odd indices.
        
        return interleaved
        
def wildfire_model(a,b,T,N_x,N_t,T_0,S_0,cT_a,dT_a,hT_a,cT_b,
    dT_b,hT_b,cS_a,dS_a,hS_a,cS_b,dS_b,hS_b,A,B,C1,C2,nu):
    # a - float
    # b - float, a < b
    # T - positive float
    # N_x - positive integer, N_x > 2, N_x = number of mesh nodes in x
    # N_t - positive integer, N_t > 1, N_t = number of mesh nodes in t
    # T_0 - function handle for the initial function auxiliary condition for T
    # S_0 - function handle for the initial function auxiliary condition for S
    # cT_a - function handle
    # dT_a - function handle
    # hT_a - function handle
    # cT_b - function handle
    # dT_b - function handle
    # hT_b - function handle
    # cS_a - function handle
    # dS_a - function handle
    # hS_a - function handle
    # cS_b - function handle
    # dS_b - function handle
    # hS_b - function handle
    # T - a two dimensional numpy array containing floats.
    # Rows correspond to time and columns to x for T.
    # S - a two dimensional numpy array containing floats.
    # Rows correspond to time and columns to x for S.
    
    wfSolver = wildfireSolver(a,b,T,N_x,N_t,T_0,S_0,cT_a,dT_a,hT_a,cT_b,
    dT_b,hT_b,cS_a,dS_a,hS_a,cS_b,dS_b,hS_b,A,B,C1,C2,nu)
    return wfSolver.solve()
    
