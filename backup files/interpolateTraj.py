import numpy as np
#given waypoints in wp, will linearly interpolate between
#them, with segments number of points in between each waypoint
# (including the waypoints themselves)
#wp: np.array([[x1 y1 z1 rx1 ry1 rz1],
#               [x2 y2 z2 rx1 ry2 rz2],...])
#segments: int>=2, the number of points to interpolate between each waypoint(inclusive)
def interpolateTraj(wp,segments):
    numWp=len(wp)
    dimWp=len(wp[0])
    interpolatedWp=np.zeros(((segments-1)*(numWp-1)+1,6))#[[0]*6 for i in range(numWp)]
    for i in range(numWp-1):
        subArray=np.array([np.linspace(wp[i][j],wp[i+1][j],segments)
                             for j in range(dimWp)])
        subStart=(segments-1)*i
        subEnd=(segments-1)*(i+1)+1
        interpolatedWp[subStart:subEnd,0:6]=np.transpose(subArray)
    return interpolatedWp