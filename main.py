import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# The gravitational constant in AU solar masses (G*M)
G = 4*np.pi**2
# The masses and initial positions of the Sun and other planets in AU and solar masses
m_sun = 1
m_venus = 0.000002447838
m_earth = 0.000003003489
m_mars = 0.0000003227151
m_jupiter = 0.00095479194
m_mercury=0.000000165
x_sun = np.array([0, 0, 0])
x_venus = np.array([0.723332, 0, 0])
x_earth = np.array([1, 0, 0])
x_mars = np.array([1.523679, 0, 0])
x_jupiter = np.array([5.2026, 0, 0])

# The function that computes the derivative of the state vector
def f(t, r):
    lam=1.1e-8
    r_sun = r - x_sun
    r_venus = r - x_venus
    r_earth = r - x_earth
    r_mars = r - x_mars
    r_jupiter = r - x_jupiter
    rn_sun = np.linalg.norm(r_sun)
    rn_venus = np.linalg.norm(r_venus)
    rn_earth = np.linalg.norm(r_earth)
    rn_mars = np.linalg.norm(r_mars)
    rn_jupiter = np.linalg.norm(r_jupiter)
    a_sun = (-G*m_sun*r_sun/rn_sun**3)*(1+(lam/rn_sun**2))
    a_venus = (-G*m_venus*r_venus/rn_venus**3)*(1+(lam/rn_venus**2))
    a_earth = (-G*m_earth*r_earth/rn_earth**3)*(1+(lam/rn_earth**2))
    a_mars = (-G*m_mars*r_mars/rn_mars)**3*(1+(lam/rn_mars**2))
    a_jupiter = (-G*m_jupiter*r_jupiter/rn_jupiter**3)*(1+(lam/rn_jupiter**2))
    
    a = a_sun + a_venus + a_earth + a_mars + a_jupiter
    
    return a
    
def leapfrog_tt(f, r0, v0, t0, w0, h):

    hw = h/(2.0*w0)
    # half h/2w0
    t1=t0+hw
    r1=r0+hw*v0
    #id=0, get r1/2
    r2=np.dot(r1, r1)
    #get r2=x*x + y*y +z*z
    r12 = np.sqrt(r2)
    #r12
    #2nd step: calc v1 using r at h/2
    v1 = v0+ h*r12*f( t1,r1)
    # id=1 for g(r) at h/2
    rdotv = np.dot(r1, v0+v1)/2
    #r.v1/2
    w1=w0-rdotv*h/r2
    #w0-r.(v1/2)*h/r^2
    #3rd step: calc r by another 1.2 step using v1
    hw=h/(2.0*w1)
    t1=t1+hw
    r1=r1+hw*v1
    #get r1 at t+h
    return r1, v1, t1, w1

# The initial conditions

# Initial position vector in AU
r0 = np.array([0.3075, 0, 0])
# Initial velocity vector in AU/year
v0 = np.array([0, 12.44, 0])
# Initial time in years
t0 = 0
# Initial value of auxiliary variable W
w0 = 1 / np.linalg.norm(r0)
# Time step in years
h = 1e-4
# Simulate Mercury's orbit
t = t0
r = r0
v = v0
w = w0
positions = [r]
vel=[v]
time=[t]
for i in range(int(100 / h)):
    r, v, t, w = leapfrog_tt(f, r, v, t, w, h)
    positions.append(r)
    vel.append(v)
    time.append(t)

# Plot Mercury's orbit
positions = np.array(positions)
plt.plot(positions[:, 0], positions[:, 1])
plt.xlabel('X coordinates')
plt.ylabel('Y coordinates')
plt.title('There is slight shift of mercurys orbit in 40 years. Zoom in to see')
plt.show()
plt.gca().set_aspect('equal')

def rLv(m_mercury,v,r,M_sun):
    k = G*(M_sun)*m_mercury
    p = m_mercury *v
    L = np.cross(r,p)
    t = m_mercury *k*r
    runge = np.cross(p,L) - m_mercury*k*(r)
    return runge

fin=[]
for i in range(len(positions)):
    x=rLv(m_mercury,vel[i],positions[i],m_sun)
    fin.append(x)

list = []
for i in range(len(fin)):
    x1= np.arctan(fin[i][1]/fin[i][0])
    list.append(x1)

plt.plot(time,list,color = 'b')
plt.xlabel('Time in years')
plt.ylabel('Precession')
plt.title('Mercury orbit')
plt.show()