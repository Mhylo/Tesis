# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:02:44 2020

@author: Wanli Zhao
"""


import cupy as cp
import numpy as np
from MatrixDft import MatrixDftGPU
from MatrixDft import MatrixDftCPU
import sys
class Gauss_beam():
    
    def __init__(self,L0,M,N,lamb,w0):
        self.L0 = L0
        self.M = M
        self.N = N
        self.lamb = lamb
        self.w0 = w0
        self.delta = L0/(self.N-1)
        self.k = 2*cp.pi/self.lamb
        self.x = cp.linspace(-self.N/2,self.N/2-1,self.N)*self.delta
        self.y = cp.linspace(-self.M/2,self.M/2-1,self.M)*self.delta
        
    def generate_beam(self,Phase):
        xx,yy = cp.meshgrid(self.x,self.y)
        Wz0 = self.w0
        A0 = self.w0/Wz0*cp.exp(-(cp.power(xx,2)+cp.power(yy,2))/cp.power(Wz0,2))
        U0 = A0*cp.exp(1j*Phase)
        return U0

  
    def peaks(self,M,N):
        x = cp.linspace(-N/2,N/2-1,N)*self.delta
        y = cp.linspace(-M/2,M/2-1,M)*self.delta
        xx,yy = cp.meshgrid(x,y)
        Phase = 3*cp.power((1-xx),2)*cp.exp(-cp.power(xx,2)-cp.power((yy+1),2))\
            -10*(xx/5-cp.power(xx,3)-cp.power(yy,5))*cp.exp(-cp.power(xx,2)-cp.power(yy,2))\
                -(1/3)*cp.exp(-cp.power(xx+1,2)-cp.power(yy,2))
        return Phase

    
class LightPropagate():
    
     def __init__(self,lamb,delta,M,N):
         self.lamb = lamb
         self.delta = delta
         self.k = 2*cp.pi/self.lamb 
         self.M = M
         self.N = N
        
     def propagate_gpu(self,U0,z,k1,k2,k3,k4):
        M1 = k1*self.M
        N1 = k1*self.N
        LX = self.delta*N1
        LY = self.delta*M1
        u = self.lamb/LX*cp.linspace(-N1/2,N1/2-1,N1)/k2
        v = self.lamb/LY*cp.linspace(-M1/2,M1/2-1,M1)/k2
        uu,vv = cp.meshgrid(u,v)
        H = cp.exp(1j*self.k*z*cp.sqrt(1-cp.power(uu,2)-cp.power(vv,2)))
        mad = MatrixDftGPU(U0,self.delta,k1,k2)
        Fu = mad.mdft()
        Uz = mad.midft(Fu*H,k3,k4)
        return Uz
  # ASM based on FFT in GPU
     def propagate_fft(self,U0,z):   
        M = self.M
        N = self.N
        LX = self.delta*N
        LY = self.delta*M
        u = self.lamb/LX*cp.linspace(-N/2,N/2-1,N)
        v = self.lamb/LY*cp.linspace(-M/2,M/2-1,M)
        uu,vv = cp.meshgrid(u,v)
        Fu = cp.fft.fft2(U0)
        Fu = cp.fft.fftshift(Fu)
        H = cp.exp(1j*self.k*z*cp.sqrt(1-cp.power(uu,2)-cp.power(vv,2)))
        Uz = cp.fft.ifft2(cp.fft.ifftshift(Fu*H))
        return Uz
  #MPASM in GPU
     def propagate_pro_GPU(self,U0,z,k1,k3,k4):
        N = self.N*k1
        if z!=0:
            fmax = cp.sqrt(cp.sqrt(cp.power(N*self.lamb,4)*+cp.power(8*N*self.lamb*z,2))\
                       -cp.power(N*self.lamb,2))/(4*cp.sqrt(2)*z*self.lamb)
            k2 = 1/(2*self.delta)/fmax
            if k2<=1:
                k2 = 1
        else:
            k2 = 1
        Uz = self.propagate_gpu(U0, z, k1, k2, k3, k4)
        return Uz
    
   # ASM based on FFT in CPU
     def propagate_fft_numpy(self,U0,z):
         M = self.M
         N = self.N
         LX = self.delta*N
         LY = self.delta*M
         u = self.lamb/LX*np.linspace(-N/2,N/2-1,N)
         v = self.lamb/LY*np.linspace(-M/2,M/2-1,M)
         uu,vv = np.meshgrid(u,v)
         H = np.exp(1j*self.k*z*np.sqrt(1-np.power(uu,2)-np.power(vv,2)))
         Fu = np.fft.fft2(U0)
         Fu = np.fft.fftshift(Fu)
         Uz = np.fft.ifft2(np.fft.ifftshift(Fu*H))
         return Uz

     def propagate_cpu(self,U0,z,k1,k2,k3,k4):
        M1 = k1*self.M
        N1 = k1*self.N
        LX = self.delta*N1
        LY = self.delta*M1
        u = self.lamb/LX*np.linspace(-N1/2,N1/2-1,N1)/k2
        v = self.lamb/LY*np.linspace(-M1/2,M1/2-1,M1)/k2
        uu,vv = np.meshgrid(u,v)
        H = np.exp(1j*self.k*z*np.sqrt(1-np.power(uu,2)-np.power(vv,2)))
        mad = MatrixDftCPU(U0,self.delta,k1,k2)
        Fu = mad.mdft()
        Uz = mad.midft(Fu*H,k3,k4)
        return Uz
    
    
    
    
    #MPASM in CPU
     def propagate_pro_CPU(self,U0,z,k1,k3,k4):
        N = self.N*k1
        if z!=0:
            fmax = np.sqrt(np.sqrt(np.power(N*self.lamb,4)*+np.power(8*N*self.lamb*z,2))\
                       -np.power(N*self.lamb,2))/(4*np.sqrt(2)*z*self.lamb)
            k2 = 1/(2*self.delta)/fmax
            if k2<=1:
                k2 = 1
        else:
            k2 = 1
        Uz = self.propagate_cpu(U0, z, k1, k2, k3, k4)
        return Uz
     def propagateProS(self,U0,Z,k1,k3,k4):       
         M1 = k1*self.M
         N1 = k1*self.N
         M2 = k3*self.M
         N2 = k3*self.N
         x = cp.reshape(cp.linspace(-self.N/2,self.N/2-1,self.N)*self.delta,(self.N,1))
         y = cp.reshape(cp.linspace(-self.M/2,self.M/2-1,self.M)*self.delta,(1,self.M))
         Lx = self.delta*N1
         Ly = self.delta*M1
         x1 = cp.reshape(cp.linspace(-k3*self.N/2,k3*self.N/2-1,k3*self.N)*self.delta*k4,(1,k3*self.N))
         y1 = cp.reshape(cp.linspace(-k3*self.M/2,k3*self.M/2-1,k3*self.M)*self.delta*k4,(k3*self.M,1))
         U = cp.zeros((M2,N2,len(Z)))
         t = 0
         for z in Z:
             if z!=0:
                 fmax = cp.sqrt(cp.sqrt(cp.power(N1*self.lamb,4)*+cp.power(8*N1*self.lamb*z,2))\
                           -cp.power(N1*self.lamb,2))/(4*cp.sqrt(2)*z*self.lamb)
                 k2 = 1/(2*self.delta)/fmax
                 if k2<=1:
                     k2 = 1
             else:
                 k2 = 1
             fx = cp.reshape(cp.linspace(-N1/2,N1/2-1,N1)/Lx/k2,(1,N1))
             fy = cp.reshape(cp.linspace(-M1/2,M1/2-1,M1)/Ly/k2,(M1,1))
             Mx = cp.exp(-2*cp.pi*1j*cp.dot(x,fx))
             My = cp.exp(-2*cp.pi*1j*cp.dot(fy,y))
             FU0 = cp.dot(cp.dot(My,U0),Mx)/(M1*N1)/cp.power(k2,2)
             u = self.lamb/Lx*cp.linspace(-N1/2,N1/2-1,N1)/k2
             v = self.lamb/Ly*cp.linspace(-M1/2,M1/2-1,M1)/k2
             uu,vv = cp.meshgrid(u,v)
             H = cp.exp(1j*self.k*z*cp.sqrt(1-cp.power(uu,2)-cp.power(vv,2)))
             Mx1 = cp.exp(2*cp.pi*1j*cp.dot(fx.T,x1))
             My1 = cp.exp(2*cp.pi*1j*cp.dot(y1,fy.T))
             Uz = cp.dot(cp.dot(My1,FU0*H),Mx1)
             U[:,:,t] = Uz
             t+=1
         return U