import torch
import numpy as np
from matplotlib import pyplot as plt

from pops_solver import *



##################################################################
#
#  Read in the array and pattern definitions 
#
##################################################################

pattern_region_lim = 0.1

objective_beam_footprint_img = setup_img_file('GT.png')
plt.figure()
plt.ion()
plt.imshow( objective_beam_footprint_img, cmap='bone')
plt.colorbar()

plt.figure()
U,Up = steering_and_penalty_vec_setup_from_img_file('GT.png',pattern_region_lim, pattern_region_lim)
U[1,:] = -U[1,:]
Up[1,:] = -Up[1,:]
plt.plot(U[0,:] * U[2,:], U[1,:] * U[2,:],'+')


array_layout_img = setup_img_file('Buzz_array_layout.png')
plt.figure()
plt.imshow( array_layout_img , cmap='bone'); plt.colorbar()


plt.figure()
P,N,X,Y = arrayvar_setup_from_img_file('Buzz_array_layout.png')
P[1,:] = -P[1,:]
plt.plot(P[1,:], P[0,:],'+')

##################################################################
#
#  Set up the ML Model 
#
##################################################################

propagation_phases = 2 * np.pi * np.matmul( U.transpose(), P)
F = np.exp( 1j * propagation_phases )

propagation_phases_pen = 2 * np.pi * np.matmul( Up.transpose(), P)
Fpen = np.exp( 1j * propagation_phases_pen )

(ndim, nP) = P.shape
phi = torch.tensor( 0.9 * np.random.randn(nP,1) , requires_grad=True)
phiSGD = torch.tensor( 0.9 * np.random.randn(nP,1) , requires_grad=True)


model = POPS_NN(F,Fpen)
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
nEpoch = 250

for epoch in range(nEpoch):
   iter_scale = (1 + 0*epoch/nEpoch)
   for idx in range(nSGD):
      E, Epen = model.forward(phi)
      loss = -iter_scale * torch.mean(E)/norm_factor + torch.mean(Epen)/norm_factor + torch.mean( torch.abs(  0.05 - E/norm_factor))
      optimSGD.zero_grad()
      loss.backward()
      optimSGD.step()
      prn_count += 1
      if prn_count > nSGD -1:
         print('SGD -- % done:', epoch/nEpoch, ' -- ',  loss)
         prn_count = 0

   for idx in range(nAdam):
      E, Epen = model.forward(phi)
      loss = - iter_scale * torch.mean(E)/norm_factor + torch.mean(Epen)/norm_factor + torch.mean( torch.abs(  0.05 - E/norm_factor))
      optim.zero_grad()
      loss.backward()
      optim.step()
      prn_count += 1
      if prn_count > nAdam -1:
         print('Adam -- % done:', epoch/nEpoch, ' -- ',  loss)
         prn_count = 0
   

##################################################################
#
#  Plot some things
#
##################################################################

npts = 151
Uplot = steering_vec_setup(npts,npts,4 * pattern_region_lim, 4 * pattern_region_lim )
Fplot = beamforming_matrix_setup(P,Uplot)
v = np.exp(1j*phi.detach().numpy())

Efield = np.matmul(Fplot, v)

pattern = np.abs(Efield)**2


plt.figure()
plt.imshow( np.flipud( np.reshape(pattern, (npts,npts) )) )
plt.colorbar()


plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D( U[1,:], U[0,:], E.detach().numpy() )
