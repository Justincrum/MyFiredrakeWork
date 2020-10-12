
#Creating a script to graph the results from the mixed poisson problem in 2d.
import csv
from matplotlib import pyplot as plt
import numpy as np
from numpy import genfromtxt

Sminus = genfromtxt('2d-results-primal-poisson-Serendipity.csv', delimiter=',')
RTC = genfromtxt('2d-results-primal-poisson-Lagrange.csv', delimiter=',')
#Sminus = genfromtxt('Sminus.csv', delimiter=',')
#RTC = genfromtxt('RTC.csv', delimiter=',')


S2 = Sminus[0:5, 1:4]
S3 = Sminus[5:10, 1:4]
S4 = Sminus[10:15, 1:4]
S5 = Sminus[15:20, 1:4]
S6 = Sminus[20:25, 1:4]

""" S1 = Sminus[0:5, 1:5]
S2 = Sminus[5:10, 1:5]
S3 = Sminus[10:15, 1:5]
S4 = Sminus[15:20, 1:5]
S5 = Sminus[20:25, 1:5]
S6 = Sminus[25:30, 1:5] """

R2 = RTC[0:5, 1:4]
R3 = RTC[5:10, 1:4]
R4 = RTC[10:15, 1:4]
R5 = RTC[15:20, 1:4]
R6 = RTC[20:25, 1:4]

""" R1 = RTC[0:5, 1:5]
R2 = RTC[5:10, 1:5]
R3 = RTC[10:15, 1:5]
R4 = RTC[15:20, 1:5]
R5 = RTC[20:25, 1:5]
R6 = RTC[25:30, 1:5] """

plt.title('Trimmed Serendipity and Tensor Elements N vs Error')
plt.xlabel('N')
plt.ylabel('Error in U')
plt.loglog(S2[:,0], S2[:,2], 'o-b', label='Sminus2')
plt.loglog(S3[:,0], S3[:,2], 'o-y', label='Sminus3')
plt.loglog(S4[:,0], S4[:,2], 'o-r', label='Sminus4')
plt.loglog(S5[:,0], S5[:,2], 'o-c', label='Sminus5')
plt.loglog(S6[:,0], S6[:,2], 'o-g', label='Sminus6')

plt.loglog(R2[:,0], R2[:,2], 'o:b', label='RTCF2')
plt.loglog(R3[:,0], R3[:,2], 'o:y', label='RTCF3')
plt.loglog(R4[:,0], R4[:,2], 'o:r', label='RTCF4')
plt.loglog(R5[:,0], R5[:,2], 'o:c', label='RTCF5')
plt.loglog(R6[:,0], R6[:,2], 'o:g', label='RTCF6')

plt.legend()
plt.show()

