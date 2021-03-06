import matplotlib
#matplotlib.use('webagg')

from scipy.special import binom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
def Bernstein(n, k):
    coeff = binom(n, k)

    def _bpoly(x):
        return coeff * x ** k * (1 - x) ** (n - k)

    return _bpoly


def Bezier(points, num=1000):
    N = len(points)
    t = np.linspace(0,1, num=num)
    curve = np.zeros((num, 2))

    for ii in range(N):
        curve += np.outer(Bernstein(N-1, ii)(t), points[ii])
    
    return curve

def _build_bezier(points):
    x,y = Bezier(points).T
    return x,y
def _build_center(axis, perpLine, curve):
    x,y,r,v = calcCenterPoints(axis, perpLine, curve).T
    return x,y
def _build_left_vel(p):
    x = p[:,0,2].T #Length of bezier curve
    l = p[:,0,0].T #Left wheel velocity
    r = p[:,0,1].T 

    scaleX = np.amax(x) # Normalize Bezier length
    scaleY = np.amax((l)) # Normalize velocity between 0 and 1
    #print("{0}, {1}".format(x,y))
    return (x/scaleX),(l+r)

def _build_right_vel(p):
    x = p[:,0,2].T # Length of bezier
    y = p[:,0,1].T # Right wheel velocity

    scaleX = np.amax(x)
    scaleY = np.amax(y)

    return (x/scaleX),(y)

def deriv(curve):
    derivline = np.zeros((curve.shape[0]-1, 2))
    for i in range(0, curve.shape[0]-1):
        dx = curve[i+1][0] - curve[i][0]
        dy = curve[i+1][1] - curve[i][1]

        derivline[i][0] = curve[i][0] + (dx/2)
        derivline[i][1] = (dy/dx)

    return derivline

def calcPerp(curve):
    perpline = np.zeros((curve.shape[0]-1, 2))
    for i in range(0, curve.shape[0]-1):
        dx = curve[i+1][0] - curve[i][0]
        dy = curve[i+1][1] - curve[i][1]

        perpline[i][0] = curve[i][0] + (dx/2)
        perpline[i][1] = -(dx/dy)
    return perpline  

def calcCenterPoints(axis, perpline, curve):
    centerpointline = np.zeros((perpline.shape[0]-1,4))
    for i in range(0, perpline.shape[0]-1):
        if perpline[i][1] == perpline[i+1][1]:
            print("Parallel ortho found")
            centerpointline[i][0] = 0
            centerpointline[i][1] = 0
            centerpointline[i][2] = 0
            dx = curve[i+1][0] - curve[i][0]
            dy = curve[i+1][1] - curve[i][1]

            centerpointline[i][3] = (dx**2+dy**2)**0.5

        else:
            currx = curve[i][0]
            curry = curve[i][1]
            dx = curve[i+1][0]-currx
            dy = curve[i+1][1]-curry
            numerator = -currx*perpline[i+1][1] -dx*perpline[i+1][1] + curve[i+1][1] - curve[i][1] + currx*perpline[i]
            denom = perpline[i][1] - perpline[i+1][1]

            cx = numerator/denom
            cy = perpline[i][1] * (cx-currx) + curry
            centerx = cx[0]
            centery = cy[0]
            #print("{0}, {1}".format(centerx,centery))
            centerpointline[i][0] = centerx
            centerpointline[i][1] = centery
            
            r = ((centerx- currx)**2 + (centery-curry)**2)**0.5
            centerpointline[i][2] = r
            centerpointline[i][3] = (dx**2 + dy**2)**0.5
            #print("RADIUS {0}".format(r))
            if i%5 == 0:
                circle = plt.Circle((centerx, centery), r, color='r', fill=False)
                axis.add_artist(circle)
            #print(r)
    return centerpointline

def calcLeftRightVel(centerpointline, time):
    leftRightVel = np.zeros((centerpointline.shape[0], 1, 4))
    actualTotalLen = np.sum(centerpointline[:,3])
    #print(actualTotalLen)
    totalLen = 0
    for i in range(0, centerpointline.shape[0]):
        #print(centerpointline[i][3])
        totalLen = totalLen + centerpointline[i][3]
        if(centerpointline[i][2] > 1000):
            print("ERROR")
            leftRightVel[i][0][0] = 1
            leftRightVel[i][0][1] = 1
            leftRightVel[i][0][2] = totalLen
            leftRightVel[i][0][3] = centerpointline[i][3]/actualTotalLen
        else:
            dt = (centerpointline[i][3]/actualTotalLen)*time
            print(dt)
            angularVel = (centerpointline[i][3]/dt) /centerpointline[i][2]
            vl = angularVel * (centerpointline[i][2]+1.0)
            vr = angularVel * (centerpointline[i][2]-1.0)
            leftRightVel[i][0][0] = vl
            leftRightVel[i][0][1] = vr
            leftRightVel[i][0][2] = totalLen
            leftRightVel[i][0][3] = centerpointline[i][3]/actualTotalLen

    return leftRightVel


points = [[0,0], [0,1], [5,5], [5,0],[6,3]]

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))

line = Line2D([], [], c='#666666')
lineCenter = Line2D([], [], c='#22FF22')
lineLeftVel = Line2D([], [], c='#0000FF')
lineRightVel = Line2D([], [], c='#FF0000')

beziercurve = ax1.add_line(line)
centercurve = ax1.add_line(lineCenter)
leftCurve = ax2.add_line(lineLeftVel)
rightCurve = ax2.add_line(lineRightVel)

beziercurve.set_data(*_build_bezier(points))
bez = Bezier(points)
perp = calcPerp(bez)

leftRightVel = calcLeftRightVel(calcCenterPoints(ax1,perp,bez),5)

centercurve.set_data(*_build_center(ax1, perp, bez))


leftCurve.set_data(*_build_left_vel(leftRightVel))
rightCurve.set_data(*_build_right_vel(leftRightVel))
ax1.set_xlim(-5,5)
ax1.set_ylim(-5,5)
ax2.set_xlim(0,5)
ax2.set_ylim(0,5)

#ax1.add_line(line)


#print(Bezier(points))
plt.show()
