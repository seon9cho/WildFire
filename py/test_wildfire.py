from wildfire_model import wildfire_model
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

a = 0
b = 10
T = 20
N_x = 100
N_t = 400
x = np.linspace(a, b, N_x)

#have most of the heat be at the origin.
T_0 = lambda x : 4* np.exp(-x**2)

#Distribute fuel uniformly.
S_0 = lambda x : np.full_like(x, 4)

#Insulate T at the ends.
hT_a = hT_b = lambda x : np.zeros_like(x)
cT_a = cT_b = lambda x : np.zeros_like(x)
dT_a = dT_b = lambda x : np.ones_like(x)

#Put no fuel at the ends.
hS_a = hS_b = lambda x : np.zeros_like(x)
cS_a = cS_b = lambda x : np.ones_like(x)
dS_a = dS_b = lambda x : np.zeros_like(x)

#No idea what to set the constants to, so I'm just going to set them all to 1.
A = B = C1 = C2 = nu = 1

#Have the wildfire model solve it.
print("Calling wildfire model")
T, S = wildfire_model(a,b,T,N_x,N_t,T_0,S_0,cT_a,dT_a,hT_a,cT_b,
                        dT_b,hT_b,cS_a,dS_a,hS_a,cS_b,dS_b,hS_b,A,B,C1,C2,nu)
print("Back from wildfire model")
                        
#Animate the solution
#Make figure and axis
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim((0, 10))
ax.set_ylim((0, 5))

#Line objects.
temp_line, = ax.plot([],[],'r',label='T')
fuel_line, = ax.plot([],[],'g',label='S')

#update func.
def update(i):
    temp_line.set_data(x, T[i, :])
    fuel_line.set_data(x, S[i, :])
    return temp_line, fuel_line

#Animation object.
print("Constructing animation")
plt.legend()
ani = animation.FuncAnimation(fig, update, frames=N_t,interval=25)
#plt.show()
ani.save("wildfire.mp4")

print("Animation saved")
