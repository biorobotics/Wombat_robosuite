import matplotlib.pyplot as plt
from math import sin, cos, pi, tan, radians, degrees
from numpy import ones,vstack
from numpy.linalg import lstsq
import numpy as np
import plotly.express as px
from itertools import chain

class LameCurve:
    """
    Lame Curve Trajectory for a parallel robot given start and end position of end effector
    """
    def __init__(self, pickHeight = 5, horizontalSpan = 10):
        self.pH = pickHeight
        self.hS = horizontalSpan
        self.e = self.pH/2
        self.f = self.hS/2 - 3
    def equation_of_line(self,points):
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        return m,c
    def curve_coordinates(self, theta):
        # print("&&&&&&",(2*self.e - self.hS)*(1 + tan(theta)**3)**(1/3))
        tanTerm = (1+tan(theta)**3)**(1/3)
        u = (2*self.e + (2*self.e - self.hS)*tanTerm) / 2*tanTerm
        w = (self.f*tan(theta) + (self.pH - self.f)*tanTerm) / tanTerm
        # print("U , W = ",u, w)
        return u,w
    def construct_curve(self, start, end):
        x,y,z = [],[],[]

        #Line AB
        AB = self.pH - self.f
        xAB = [start[0]]*10
        yAB = [start[1]]*10
        zAB = np.linspace(start[2],start[2]+AB,10)

        #Curve BC
        BC_coords = []
        for i in range(1,60):  #Rotate curve from 0 to 60 degrees
            u,w = self.curve_coordinates(radians(i))
            BC_coords.append([u,w])          #translation
        #Tranform 2D to 3D
        xBC, zBC = np.array(BC_coords).T
        yBC = np.zeros(len(xBC))
        xBC_Shift = xBC[0]
        for i in range(len(xBC)):
            xBC[i] = xBC[i] - xBC_Shift
            xBC[i] += xAB[-1]
            yBC[i] += yAB[-1]
            # zBC[i] += zAB[-1]

        #Line CF
        CF = self.hS + 2*(xBC[-1]-xBC[0])
        # print("CF = ",CF)
        xCF = np.linspace(0,CF,10)
        yCF = [yBC[-1]]*10
        zCF = [zBC[-1]]*10
        for i in range(len(xCF)):
            xCF[i] = -xCF[i] + xBC[-1]
        # print("CF = ",xCF)

        #Curve FG
        FG_coords = []
        for i in reversed(range(1,60)):  #Rotate curve from 0 to 60 degrees
            u,w = self.curve_coordinates(radians(i))
            FG_coords.append([u,w])
        # print("FG_Coords = ",xCF[-1],FG_coords)
        #Tranform 2D to 3D
        xFG,zFG = np.array(FG_coords).T
        yFG = np.zeros(len(xFG))
        # print("XFG = ",xFG)
        # print("Last x in CF = ",xCF[-1])
        xFG_M = xFG[0]
        for i in range(len(xBC)):
            xFG[i] = xFG_M-xFG[i] + xCF[-1]
            # print("XFG = ",xFG[i])
            yFG[i] += yCF[-1]
            # zFG[i] += zAB[-1]

        #Line GH
        GH = self.pH - self.f
        xGH = [xFG[-1]]*10
        yGH = [yFG[-1]]*10
        zGH = np.linspace(zAB[-1],zAB[-1]-GH,10)

        #combine and plot
        x = list(chain(xAB,xBC,xCF,xFG,xGH))
        y = list(chain(yAB,yBC,yCF,yFG,yGH))
        z = list(chain(zAB,zBC,zCF,zFG,zGH))
        i = list(range(len(x)))
        x = np.array(x)
        x = 0.15*x/10
        x = list(x)
        z = np.array(z)
        z = z*0.05/5 + 0.65
        z = list(z)

        #Plotly Plot
        fig = px.scatter_3d(x=x, y=y, z=z, color=i)
        fig.show()
        # ax = plt.axes(projection='3d')
        # ax.plot3D(x,y,z)
        # plt.title("Curve")
        # plt.show()
        return np.array(x), np.array(y), np.array(z)

lc = LameCurve()
x,y,z = lc.construct_curve(np.array([0,0,0.65]),[-0.15,0,0.65])
# print(len(x))