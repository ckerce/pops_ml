'''
   Copyright (C) 2021 by the Georgia Tech Research Institute

   About:  POPS_ML provides a reference implementation to illustrate the use of
   machine learning APIs in the development of optimization problems.  As such,
   it requires only the PyTorch and numpy libraries to implement.  Existing complex
   linear algebra libraries for PyTorch or Tensorflow would allow for a slightly 
   more concise implementation.

   License: POPS_ML is free software: you can redistribute it and/or modify it 
   under the terms of the GNU Lesser General Public License (Version 3) as published by the
   Free Software Foundataion.

   Citation: Please cite this work as
   J. Clayton Kerce, A Phase Only Pattern Synthesis Reference Implementation as an Adversairal 
   Machine Learning Model in PyTorch

   @author Clayton Kerce <clayton.kerce@gtri.gatech.edu>
   @date   30 April 2021
'''
import torch
import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt


def setup_img_file(filename):
   img = image.imread(filename)
   if len(img.shape) > 2:
      img = np.average(img,2)
      img = np.round(img)
      #img = 1. - img
   return img


def arrayvar_setup_from_img_file(filename):
   img = setup_img_file(filename) 
   A = arrayvar_setup_from_mask(img)
   return A

def arrayvar_setup_from_mask(M, clip_val=0.1):
   '''
       Unit-ed quantities are scaled to center frequency wavelength units
       P = list of element positions in 3D (\labmda scaling)
       Mmax = mask indicating where to maximize energy
       Mmin = mask indicating where to minimize energy
       N = list of surface normals 
       
   '''
   (nx,ny) = M.shape 
   num_el = nx * ny
   P = np.zeros((3,num_el))
   N = np.zeros((3,num_el))
   X = np.zeros((nx,ny))
   Y = np.zeros((nx,ny))
   
   count = 0
   scale_factor = 0.35 #0.5
   for xidx in range(nx):
      for yidx in range(ny):
         if M[xidx,yidx] > clip_val:
            xtmp = scale_factor * ((nx-1)/4. - xidx/2.)
            ytmp = scale_factor * ((ny-1)/4. - yidx/2.)
            P[1,count] = xtmp 
            P[0,count] = ytmp 
            X[xidx,yidx] = xtmp
            Y[xidx,yidx] = ytmp

            N[2,count] = 1.
            count += 1
   P = P[:,0:count] 
   N = N[:,0:count] 
   return P,N,X,Y 

def steering_and_penalty_vec_setup_from_img_file(filename, uxlim=0.5, uylim=0.5):
   img = setup_img_file(filename)
   U = steering_setup_from_mask(img, uxlim, uylim, oversample=2)
   Upen = steering_setup_from_mask(1-img, uxlim, uylim)
   return U, Upen

def steering_vec_setup_from_img_file(filename, uxlim=0.5, uylim=0.5):
   img = setup_img_file(filename)
   U = steering_setup_from_mask(img, uxlim, uylim)
   return U
    
def steering_setup_from_mask(img, uxlim=0.5, uylim=0.5, oversample=1):
   tmpn = oversample * np.array(img.shape) * [uxlim, uylim]
   (nUx,nUy) = tmpn.astype(int) 
   print(nUx, nUy)
   
   U = steering_vec_setup(nUx,nUy, uxlim, uylim, img, oversample)
   return U
   
def steering_vec_setup(nUx, nUy, uxlim=0.5, uylim=0.5, imgMask=[], oversample=1):
   '''
       U = list of pointing directions to evaluate (unit vectors)
   '''
   if len(imgMask) < 1:
      mask_test = lambda i,j: True
      U = np.zeros( (3,nUx*nUy) )
   else:
      s = float(oversample)
      mask_test = lambda i,j: imgMask[ int( i/uxlim / s), int( j/uylim/s) ] > 0.9
      numU = np.min( [float(nUx * nUy), np.sum(imgMask)])
      U = np.zeros((3, int(numU)))

   uxvals = np.linspace(-uxlim, uxlim, nUx)
   uyvals = np.linspace(-uylim, uylim, nUy)
   count = 0

   for xidx in range(nUx):
      for yidx in range(nUy):
         if mask_test(xidx,yidx):
            xtmp = uxvals[xidx] 
            ytmp = uyvals[yidx] 
            ztmp2 = 1 - xtmp**2 - ytmp**2
            if ztmp2 > 0:
              U[:,count] = np.array([ytmp,xtmp,np.sqrt(ztmp2)])    
              count += 1
   U = U[:,0:count]
   return U


def array_setup_regular_grid(nx,ny):
   '''
      Returns numpy array of element positions, P, and array surface normals, N.
      num_el = nx * ny element positions
      P = 3 x num_el
      N = 3 x num_el
   '''
   num_el = nx * ny
   P = np.zeros((3,num_el))
   N = np.zeros((3,num_el))
   X = np.zeros((nx,ny))
   Y = np.zeros((nx,ny))
   count = 0
   for xidx in range(nx):
      for yidx in range(ny):
         xtmp = (nx-1)/4. - xidx/2.
         ytmp = (ny-1)/4. - yidx/2.
         P[0,count] = xtmp 
         P[1,count] = ytmp 
         X[xidx,yidx] = xtmp
         Y[xidx,yidx] = ytmp

         N[2,count] = 1.
         count += 1

   return P,N,X,Y
    
  
def directional_gain(U,N,p):
    '''
       For use in computing element gains on conformal antennas
    '''
    return torch.pow(torch.sum(U*N,0),2)

def beamforming_matrix_setup(P,U):
   propagation_phases = 2*np.pi*np.matmul(U.transpose(), P)
   F = np.exp(1j * propagation_phases)
   return F

class POPS_NN_CPU():
   def __init__(self, F_main, F_penalty):
      '''
          Unit-ed quantities are scaled to center frequency wavelength units
          P = list of element positions in 3D (\labmda scaling)
          U = list of pointing directions to evaluate (unit vectors)
          Mmax = mask indicating where to maximize energy
          Mmin = mask indicating where to minimize energy
          N = list of surface normals

      '''
      super(POPS_NN_CPU, self).__init__()
      self.Fr = torch.tensor( np.real(F_main), requires_grad=False)
      self.Fi = torch.tensor( np.imag(F_main), requires_grad=False)
      self.Mr = torch.tensor( np.real(F_penalty), requires_grad=False)
      self.Mi = torch.tensor( np.imag(F_penalty), requires_grad=False)

   def forward(self,x):
      xr = torch.cos( x )
      xi = torch.sin( x )

      yr = torch.mm( self.Fr, xr) - torch.mm(self.Fi,xi)
      yi = torch.mm( self.Fi, xr) + torch.mm(self.Fr,xi)
      E = torch.square(yr) + torch.square(yi)


      ppr = torch.mm( self.Mr, xr) - torch.mm(self.Mi,xi)
      ppi = torch.mm( self.Mi, xr) + torch.mm(self.Mr,xi)
      Epen = torch.square(ppr) + torch.square(ppi)

      return E, Epen

           
class POPS_NN():
   def __init__(self, F_main, F_penalty, target_device='cpu'):
      '''
          Unit-ed quantities are scaled to center frequency wavelength units
          P = list of element positions in 3D (\labmda scaling)
          U = list of pointing directions to evaluate (unit vectors)
          Mmax = mask indicating where to maximize energy
          Mmin = mask indicating where to minimize energy
          N = list of surface normals 
      '''
      super(POPS_NN, self).__init__()
      if target_device == 'cuda':
         actual_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu').type
      else:
         actual_device='cpu'
      self.Fr = torch.tensor( np.real(F_main), requires_grad=False).to(actual_device) 
      self.Fi = torch.tensor( np.imag(F_main), requires_grad=False).to(actual_device) 
      self.Mr = torch.tensor( np.real(F_penalty), requires_grad=False).to(actual_device) 
      self.Mi = torch.tensor( np.imag(F_penalty), requires_grad=False).to(actual_device) 
      self.actual_device = actual_device

   def to(self, target_device='cuda'):
   
      if target_device == 'cuda':
         actual_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      else:
         actual_device='cpu'

      print(actual_device)
      self.Fr = self.Fr.to(actual_device)
      self.Fi = self.Fi.to(actual_device)
      self.Mr = self.Mr.to(actual_device)
      self.Mi = self.Mi.to(actual_device)
      
      return actual_device
      
   def forward(self,x):
      xr = torch.cos( x )
      xi = torch.sin( x )

      yr = torch.mm( self.Fr, xr) - torch.mm(self.Fi,xi)
      yi = torch.mm( self.Fi, xr) + torch.mm(self.Fr,xi)
      E = torch.square(yr) + torch.square(yi) 


      ppr = torch.mm( self.Mr, xr) - torch.mm(self.Mi,xi)
      ppi = torch.mm( self.Mi, xr) + torch.mm(self.Mr,xi)
      Epen = torch.square(ppr) + torch.square(ppi) 

      return E, Epen 
