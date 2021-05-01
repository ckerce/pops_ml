
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

class POPS_FarFieldNN():
   def __init__(self, F_main):
      '''
          Unit-ed quantities are scaled to center frequency wavelength units
          P = list of element positions in 3D (\labmda scaling)
          U = list of pointing directions to evaluate (unit vectors)
          Mmax = mask indicating where to maximize energy
          Mmin = mask indicating where to minimize energy
          N = list of surface normals 
          
      '''
      super(POPS_FarFieldNN, self).__init__()
      self.Fr = torch.tensor( np.real(F_main), requires_grad=False) 
      self.Fi = torch.tensor( np.imag(F_main), requires_grad=False) 
      
   def forward(self,x):
      xr = torch.cos( x )
      xi = torch.sin( x )
      yr = torch.mm( self.Fr, xr) - torch.mm(self.Fi,xi)
      yi = torch.mm( self.Fi, xr) + torch.mm(self.Fr,xi)
      E = torch.square(yr) + torch.square(yi) 
      return E 


           
class POPS_NN():
   def __init__(self, F_main, F_penalty):
      '''
          Unit-ed quantities are scaled to center frequency wavelength units
          P = list of element positions in 3D (\labmda scaling)
          U = list of pointing directions to evaluate (unit vectors)
          Mmax = mask indicating where to maximize energy
          Mmin = mask indicating where to minimize energy
          N = list of surface normals 
          
      '''
      super(POPS_NN, self).__init__()
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
