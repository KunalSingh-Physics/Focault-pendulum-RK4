import matplotlib.pyplot as plt

import numpy as np

g=9.8         #GRAVITATIONAL ACCELARATION OF THE PLANET
L=67          #LENGTH OF THE PENDULUM
omega= -0.3   #ANGULAR SPEED OF THE PLANET
phi= 3        #LATITUDE

a = 0
b = 200
h=float(np.abs(omega))
print(h)

tpoints = np.arange(a,b,h)
x = np.array([])
y = np.array([])
vx=np.array([])
vy=np.array([])
A = 2*omega*np.sin(phi)
A = complex(0,A)
B = g/L
a=len(tpoints)
u0 = 1
v0 = 0


def f1(u,v):
    dudt = v
    return dudt

def f2(u,v):
    dvdt = -(A*v)-(B*u)
    return dvdt

for n1 in tpoints:
    k11 = h*f1(u0,v0)
    k21 = h*f2(u0,v0)
    k12 = h*f1(u0+(0.5*k11),v0+(0.5*k21))
    k22 = h*f2(u0+(0.5*k11),v0+(0.5*k21))
    k13 = h*f1(u0+(0.5*k12),v0+(0.5*k22))
    k23 = h*f2(u0+(0.5*k12),v0+(0.5*k22))
    k14 = h*f1(u0+(k13),v0+(k23))
    k24 = h*f2(u0+(k13),v0+(k23))
    u0 += (k11+(2*k12)+(2*k13)+k14)/6
    v0 += (k21+(2*k22)+(2*k23)+k24)/6
    vx1= v0.real
    vy1 = v0.imag
    
    x1=u0.real
    y1=u0.imag
    x = np.append(x1,x)
    y = np.append(y1,y)
    vx=np.append(vx1,vx)
    vy=np.append(vy1,vy)
    
#TO PLOT THE GRAPH
#DECOMMENT THE PLOT CODES TO SEE THE PLOTS
'''
plt.figure()
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Position Graph')
plt.show()

plt.figure()
plt.plot(vx,vy)
plt.title('Velocity Graph')
plt.xlabel('vx')
plt.ylabel('vy')
plt.show()
'''
#For Creating Animation
from matplotlib.animation import FuncAnimation


fig, ax = plt.subplots()
fig.suptitle('velocity graph', fontsize=30,color='black')



ax.set(xlim=(-0.6,0.6), ylim=(-0.6,0.6))
ax.set_xlabel('vx',fontsize=20,color='black')
ax.set_ylabel('vy',fontsize=20,color='black')
line, = ax.plot([], []) 

def animate(i):
    line.set_data(vx[:i], vy[:i])
    return line,

anim = FuncAnimation(fig, animate, frames=len(vx)+1, interval=30, blit=True)
fig1, ax1 = plt.subplots()
fig1.suptitle('position graph', fontsize=30,color='black')


ax1.set(xlim=(-1.5,1.5), ylim=(-1.5,1.5))
ax1.set_xlabel(' x',fontsize=20,color='black')
ax1.set_ylabel('y',fontsize=20,color='black')
line1, = ax1.plot([], []) 

def animate1(j):
    line1.set_data(x[:j], y[:j])
    return line1,

anim1 = FuncAnimation(fig1, animate1, frames=len(x)+1, interval=30, blit=True)

##TO SAVE THE ANIMATION AS MP4 VIDEO DECOMMENT THE FOLLOWING CODE
'''
print('compiling video....')
anim.save('position graph with negative omega 0_3.mp4')
anim1.save('velocity graph with negative omega 0_3.mp4')
print('done')
'''