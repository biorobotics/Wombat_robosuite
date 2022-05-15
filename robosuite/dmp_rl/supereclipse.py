import matplotlib.pyplot as plt
from math import sin, cos, pi, tan, radians, degrees
from numpy import ones,vstack
from numpy.linalg import lstsq
import numpy as np
import plotly.express as px
from itertools import chain

class SuperEclipse:
    """
    SuperEclipse Curve Trajectory for a parallel robot given start and end position of end effector
    """
    def __init__(self, pickHeight = -0.1, horizontalSpan = 40):
        self.pH = pickHeight
        self.hS = horizontalSpan
        self.n = 3 

    def plot_curve(self,i,x,y,z):
        """
        Plot the Curve
        """
        #Matplot lib Plot
        fig = px.scatter_3d(x=x, y=y, z=z, color=i)
        fig.update_layout(
            title="Trajectory Plot",
            xaxis_title="X",
            yaxis_title="Y",
            legend_title="Legend Title",
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="RebeccaPurple"
            )
        )
        fig.show()

        #Plotly Plot

        ax = plt.axes(projection='3d')
        ax.plot3D(x,y,z)
        ax.set_xlabel("X",fontsize=14)
        ax.set_ylabel("Y",fontsize=14)
        ax.set_zlabel("Z",fontsize=14)

        plt.title("Trajectory Plot")
        plt.show()

    def construct_curve(self, start, end):

        #Initialise
        self.hS = end[0]-start[0]
        self.e = 0.4*self.hS                  #e(variable) --> horizontal width of curve 0 <= e <= hS/2
        self.f = 0.8*self.pH                  #f(variable) --> vertical height of curve  0 <= f <= pH


        #Line AB
        AB = self.pH - self.f
        xAB = [start[0]]*10
        yAB = [start[1]]*10
        zAB = np.linspace(start[2],start[2]+AB,10)

        #Curve BC
        t = np.linspace(np.pi,np.pi/2, 100)
        xBC = ((np.abs(np.cos(t))) ** (2 / self.n)) * self.e * np.sign(np.cos(t))
        zBC = ((np.abs(np.sin(t))) ** (2 / self.n)) * self.f * np.sign(np.sin(t))
        yBC = np.zeros(len(xBC))
        
        for i in range(len(xBC)):
            xBC[i] += xAB[-1] + self.e
            yBC[i] += yAB[-1]
            zBC[i] += zAB[-1]

        #Line CF
        CF = self.hS - 2*self.e
        xCF = np.linspace(0,CF,100)
        yCF = [yBC[-1]]*100
        zCF = [zBC[-1]]*100
        
        for i in range(len(xCF)):
            xCF[i] += xBC[-1]
            
        #Curve FG
        t = np.linspace(np.pi/2,0, 100)
        xFG = ((np.abs(np.cos(t))) ** (2 / self.n)) * self.e * np.sign(np.cos(t))
        zFG = ((np.abs(np.sin(t))) ** (2 / self.n)) * self.f * np.sign(np.sin(t))
        yFG = np.zeros(len(xFG))

        for i in range(len(xFG)):
            xFG[i] += xCF[-1]
            yFG[i] += yCF[-1]
            zFG[i] += zAB[-1]

        #Line GH
        xGH = [xFG[-1]]*10
        yGH = [yFG[-1]]*10
        zGH = np.linspace(zFG[-1],end[2],10)

        x = list(chain(xAB,xBC,xCF,xFG,xGH))
        y = list(chain(yAB,yBC,yCF,yFG,yGH))
        z = list(chain(zAB,zBC,zCF,zFG,zGH))
        i = list(range(len(x)))

        # self.plot_curve(i,x,y,z)              #To Plot the curve
        
        return np.array(x), np.array(y), np.array(z)

lc = SuperEclipse()
x,y,z = lc.construct_curve(np.array([0,0,0.65]),np.array([-0.15,0,0.65]))
# print(lc.construct_curve(np.array([0,0,0.65]),[-0.15,0,0.65]))

# print(len(x))