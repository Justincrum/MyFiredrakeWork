
#Creating a script to graph the results from the mixed poisson problem in 2d.
import csv
from matplotlib import pyplot as plt
import numpy as np
from numpy import genfromtxt

Sminus = genfromtxt('Sminus_poisson_timing.csv', delimiter=',')
RTC = genfromtxt('RTCF_poisson_timing.csv', delimiter=',')
#Sminus = genfromtxt('Sminus.csv', delimiter=',')
#RTC = genfromtxt('RTC.csv', delimiter=',')




S2 = Sminus[0:5, 2:4]
S3 = Sminus[5:10, 2:4]
S4 = Sminus[10:15, 2:4]
S5 = Sminus[15:20, 2:4]
S6 = Sminus[20:25, 2:4]

R2 = RTC[0:5, 2:4]
R3 = RTC[5:10, 2:4]
R4 = RTC[10:15, 2:4]
R5 = RTC[15:20, 2:4]
R6 = RTC[20:25, 2:4]

plt.title('Time Analysis of Sminus Vs RTCF Elements')
plt.xlabel('Time Required')
plt.ylabel('Error in U')
plt.loglog(S2[:,1], S2[:,0], 'o-b', label='Sminus2')
plt.loglog(S3[:,1], S3[:,0], 'o-y', label='Sminus3')
plt.loglog(S4[:,1], S4[:,0], 'o-r', label='Sminus4')
plt.loglog(S5[:,1], S5[:,0], 'o-c', label='Sminus5')
plt.loglog(S6[:,1], S6[:,0], 'o-g', label='Sminus6')

plt.loglog(R2[:,1], R2[:,0], 'o:b', label='RTCF2')
plt.loglog(R3[:,1], R3[:,0], 'o:y', label='RTCF3')
plt.loglog(R4[:,1], R4[:,0], 'o:r', label='RTCF4')
plt.loglog(R5[:,1], R5[:,0], 'o:c', label='RTCF5')
plt.loglog(R6[:,1], R6[:,0], 'o:g', label='RTCF6')

plt.legend()
plt.show()

