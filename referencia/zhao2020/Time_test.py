# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:04:08 2020

@author: Wanli Zhao
"""


import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from Light import Gauss_beam
from Light import LightPropagate
import time
from mpl_toolkits.mplot3d import Axes3D
import math 


# import pynufft as pynft
plt.close('all')
cp.clear_memo()
L0 = 5
N = 512
delta = L0/(N-1)
w0 = 1
lamb = 632.8e-6
zr = w0**2*math.pi/lamb
Phase = cp.zeros((N,N))

gb = Gauss_beam(L0,N,N,lamb,w0)
U0 = gb.generate_beam(Phase)
Phase0 = cp.asnumpy(Phase)
U1 = cp.asnumpy(U0)
I0 = U0*cp.conj(U0)
I = cp.asnumpy(I0)
I = np.real(I)
plt.figure(),plt.imshow(I,cmap='hot'),plt.title('Intensity')
plt.figure(),plt.imshow(Phase0,cmap='jet'),plt.colorbar(),plt.title('Phase')


x = np.linspace(-N/2,N/2-1,N)*delta
y = x
X,Y = np.meshgrid(x,y)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(X,Y,Phase0,cmap = 'jet'),ax.set_title('Phase in 3D')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

d = 12000
lp = LightPropagate(lamb,delta,N,N)



Uz = cp.zeros((N,N))

# MPASM in GPU
time1 =  time.perf_counter()
Uz_mpasmGpu = lp.propagate_pro_GPU(U0,d,1,1,1)
time2 =  time.perf_counter()

# MPASM in CPU
time3 =  time.perf_counter()
Uz_mpasmCpu = lp.propagate_pro_CPU(U1,d,1,1,1)
time4 =  time.perf_counter()

# FFT ANS in GPU
time5 = time.perf_counter()
Uz_fft = lp.propagate_fft(U0,d)
time6 =  time.perf_counter()
Iz_mpasmGpu = Uz_mpasmGpu*cp.conj(Uz_mpasmGpu)
Iz_mpasmGpu = cp.asnumpy(Iz_mpasmGpu)
Iz_mpasmGpu = np.real(Iz_mpasmGpu)



Iz_mpasmCpu = Uz_mpasmCpu*np.conj(Uz_mpasmCpu)
Iz_mpasmCpu = np.real(Iz_mpasmCpu)

    
Iz_fft = Uz_fft*cp.conj(Uz_fft)
Iz_fft = cp.asnumpy(Iz_fft)
Iz_fft = np.real(Iz_fft)


#FFT ASM in CPU
time7 = time.perf_counter()
Uz_num = lp.propagate_fft_numpy(U1,d)
time8 = time.perf_counter()
Iz_num = Uz_num*np.conj(Uz_num)
Iz_num = np.real(Iz_num)


plt.figure(),plt.imshow(Iz_mpasmGpu,cmap='hot'),plt.title('MPASM_GPU')
plt.figure(),plt.imshow(Iz_mpasmCpu,cmap='hot'),plt.title('MPASM_CPU')
plt.figure(),plt.imshow(Iz_fft,cmap='hot'),plt.title('FFT_ASM_GPU')
plt.figure(),plt.imshow(Iz_num,cmap='hot'),plt.title('FFT_ASM_CPU')


print('FFT_ASM_GPU cd:',time6-time5,'s')
print('MPASM_GPU cd:',time2-time1,'s')
print('FFT_ASM_CPU cd:',time8-time7,'s')
print('MPASM_CPU cd:',time4-time3,'s')

