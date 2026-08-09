# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 09:59:24 2020

@author: Wanli Zhao
"""


import cupy as cp 
import numpy as np


class MatrixDftGPU():
    
    def __init__(self,In,delta,k1,k2):
        self.In = In
        self.delta = delta
        self.k1 = k1
        self.k2 = k2
        self.N = cp.size(In,1)
        self.M = cp.size(In,0)
        self.x = cp.reshape(cp.linspace(-self.N/2,self.N/2-1,self.N)*self.delta,(self.N,1))
        self.y = cp.reshape(cp.linspace(-self.M/2,self.M/2-1,self.M)*self.delta,(1,self.M))
        self.Lx = k1*delta*self.N
        self.Ly = k1*delta*self.M
        self.fx = cp.reshape(cp.linspace(-k1*self.N/2,k1*self.N/2-1,k1*self.N)/self.Lx/k2,(1,k1*self.N))
        self.fy = cp.reshape(cp.linspace(-k1*self.M/2,k1*self.M/2-1,k1*self.M)/self.Ly/k2,(k1*self.M,1))
        
    def mdft(self):
        Mx = cp.exp(-2*cp.pi*1j*cp.dot(self.x,self.fx))
        My = cp.exp(-2*cp.pi*1j*cp.dot(self.fy,self.y))
        Out = cp.dot(cp.dot(My,self.In),Mx)/(cp.power(self.k1,2)*self.M*self.N)/cp.power(self.k2,2)
        return Out
    
    def midft(self,Fin,k3,k4):
        x1 = cp.reshape(cp.linspace(-k3*self.N/2,k3*self.N/2-1,k3*self.N)*self.delta*k4,(1,k3*self.N))
        y1 = cp.reshape(cp.linspace(-k3*self.M/2,k3*self.M/2-1,k3*self.M)*self.delta*k4,(k3*self.M,1))
        Mx1 = cp.exp(2*cp.pi*1j*cp.dot(self.fx.T,x1))
        My1 = cp.exp(2*cp.pi*1j*cp.dot(y1,self.fy.T))
        Out = cp.dot(cp.dot(My1,Fin),Mx1)
        return Out
    
class MatrixDftCPU():
    
    def __init__(self,In,delta,k1,k2):
        self.In = In
        self.delta = delta
        self.k1 = k1
        self.k2 = k2
        self.N = np.size(In,1)
        self.M = np.size(In,0)
        self.x = np.reshape(np.linspace(-self.N/2,self.N/2-1,self.N)*self.delta,(self.N,1))
        self.y = np.reshape(np.linspace(-self.M/2,self.M/2-1,self.M)*self.delta,(1,self.M))
        self.Lx = k1*delta*self.N
        self.Ly = k1*delta*self.M
        self.fx = np.reshape(np.linspace(-k1*self.N/2,k1*self.N/2-1,k1*self.N)/self.Lx/k2,(1,k1*self.N))
        self.fy = np.reshape(np.linspace(-k1*self.M/2,k1*self.M/2-1,k1*self.M)/self.Ly/k2,(k1*self.M,1))
        
    def mdft(self):
        Mx = np.exp(-2*np.pi*1j*np.dot(self.x,self.fx))
        My = np.exp(-2*np.pi*1j*np.dot(self.fy,self.y))
        Out = np.dot(np.dot(My,self.In),Mx)/(np.power(self.k1,2)*self.M*self.N)/np.power(self.k2,2)
        return Out
    
    def midft(self,Fin,k3,k4):
        x1 = np.reshape(np.linspace(-k3*self.N/2,k3*self.N/2-1,k3*self.N)*self.delta*k4,(1,k3*self.N))
        y1 = np.reshape(np.linspace(-k3*self.M/2,k3*self.M/2-1,k3*self.M)*self.delta*k4,(k3*self.M,1))
        Mx1 = np.exp(2*np.pi*1j*np.dot(self.fx.T,x1))
        My1 = np.exp(2*np.pi*1j*np.dot(y1,self.fy.T))
        Out = np.dot(np.dot(My1,Fin),Mx1)
        return Out