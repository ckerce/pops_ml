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
from matplotlib import pyplot as plt

from pops_solver import *



##################################################################
#
#  Read in the array and pattern definitions 
#
##################################################################

pattern_region_lim = 0.07


array_file_description = 'ML_array_template.png'
pattern_file_description = 'ML_beamspace_pattern.png'

objective_beam_footprint_img = setup_img_file(pattern_file_description)
plt.figure()
plt.ion()
plt.imshow( objective_beam_footprint_img, cmap='bone')
plt.colorbar()

plt.figure()
U,Up = steering_and_penalty_vec_setup_from_img_file(pattern_file_description,pattern_region_lim, pattern_region_lim)
U[1,:] = -U[1,:]
Up[1,:] = -Up[1,:]
plt.plot(U[0,:] * U[2,:], U[1,:] * U[2,:],'+')


T = 1.1 * np.sqrt(np.sum(np.square(Up[:,0]-Up[:,1])))
tmpI = []
for (uidx,up) in enumerate(Up.transpose()):
    curmax = 1e6
    for u in U.transpose():
        curmax = np.min( [curmax, np.sqrt( np.sum(np.square( u - up)))])
    if curmax > T:
        tmpI.append(uidx)
Up = Up[:,tmpI]
#plt.plot(Up[0,:] * Up[2,:], Up[1,:] * Up[2,:],'r+')



array_layout_img = setup_img_file(array_file_description)
plt.figure()
plt.imshow( array_layout_img , cmap='bone'); plt.colorbar()


plt.figure()
P,N,X,Y = arrayvar_setup_from_img_file(array_file_description)
P[0,:] = -P[0,:]
Np = len(P[0,:])
I = np.where( (np.random.rand(Np) > 0.5 ))
I = I[0]
P = P[:,I]
plt.plot(P[0,:], P[1,:],'+')

##################################################################
#
#  Set up the ML Model 
#
##################################################################

propagation_phases = 2 * np.pi * np.matmul( U.transpose(), P)
F = np.exp( 1j * propagation_phases )

propagation_phases_pen = 2 * np.pi * np.matmul( Up.transpose(), P)
Fpen = np.exp( 1j * propagation_phases_pen )

USE_GPU = True 
if USE_GPU:
   cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   device = cuda_device.type
else:
   device = 'cpu'

model = POPS_NN(F,Fpen, target_device=device)
actual_device = model.actual_device

(ndim, nP) = P.shape
phi = torch.tensor( 1.5 * np.random.randn(nP,1)).to(actual_device).detach().requires_grad_(True)
phiSGD = torch.tensor( 1.5 * np.random.randn(nP,1)).to(actual_device).detach().requires_grad_(True)


#optimSGD = torch.optim.SGD([phi], lr=9e-1, momentum=0.5)
phi
optimSGD = torch.optim.SGD([phi], lr=9e-1, momentum=0.5)
optim = torch.optim.Adam([phi],lr=0.01)

##################################################################
#
#  Run the optimization loop 
#
##################################################################

norm_factor = np.sum(array_layout_img)**2
prn_count = 0
nAdam = 50 
nSGD  = 50 
nEpoch = 50 

npts = 81
Uplot = steering_vec_setup(npts,npts,1.5 * pattern_region_lim, 1.5 * pattern_region_lim )
Fplot = beamforming_matrix_setup(P,Uplot)

v = np.exp(0.*1j*phi.detach().cpu().numpy())
Efield = np.matmul(Fplot, v)
pattern = np.abs(Efield)**2
plt.imsave('/tmp/array_baseline'+'.png', 10*np.log10(np.flipud( np.reshape(  np.max(np.array([1e-4*np.ones(pattern.shape), pattern/norm_factor]),0)    , (npts,npts) ))) , cmap='bone')

for epoch in range(nEpoch):
   for idx in range(nSGD):
      E, Epen = model.forward(phi)
      term = torch.mean( torch.abs( 0.99*torch.max(E)/norm_factor - E/norm_factor))
      loss = - torch.mean(E)/norm_factor + torch.mean(Epen)/norm_factor + term + torch.relu( - torch.mean(E)/norm_factor + 0.007) 
      optimSGD.zero_grad()
      loss.backward()
      optimSGD.step()
      prn_count += 1
      if prn_count > nSGD -1:
         print('SGD -- % done:', epoch/nEpoch, ' -- ',  loss)
         prn_count = 0

   for idx in range(nAdam):
      E, Epen = model.forward(phi)
      term = torch.mean( torch.abs( 0.99*torch.max(E)/norm_factor - E/norm_factor))
      loss = - torch.mean(E)/norm_factor + torch.mean(Epen)/norm_factor + term + torch.relu( - torch.mean(E)/norm_factor + 0.007)
      optim.zero_grad()
      loss.backward()
      optim.step()
      prn_count += 1
      if prn_count > nAdam -1:
         print('Adam -- % done:', epoch/nEpoch, ' -- ',  loss, torch.mean(E)/norm_factor, torch.max(E)/norm_factor)
         prn_count = 0

   v = np.exp(1j*phi.detach().cpu().numpy())
   Efield = np.matmul(Fplot, v)
   pattern = np.abs(Efield)**2
   plt.imsave('/tmp/array'+str(epoch)+'.png', 10*np.log10(np.flipud( np.reshape(  np.max(np.array([1e-4*np.ones(pattern.shape), pattern/norm_factor]),0)    , (npts,npts) ))) , cmap='bone')

   

##################################################################
#
#  Plot some things
#
##################################################################

npts = 81 
Uplot = steering_vec_setup(npts,npts,0.9 * pattern_region_lim, 0.9 * pattern_region_lim )
Fplot = beamforming_matrix_setup(P,Uplot)

v = np.exp(1j*phi.detach().cpu().numpy())
Efield = np.matmul(Fplot, v)
#Efield = np.matmul(Fplot, np.ones( v.shape ))
pattern = np.abs(Efield)**2


plt.figure()
plt.imshow( np.flipud( np.reshape(pattern, (npts,npts) )) )
plt.colorbar()


plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D( U[1,:], U[0,:], E.detach().cpu().numpy() )
