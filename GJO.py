import numpy as np
from scipy.special import gamma


class GJO:

  """ 

  GJO is a nature-based metaheuristic optimization algorithm that is inspired
  by the hunting behaviour of golden jackals to find the optimal value of a 
  function. The algorithm was developed by N. Chopra and M. Mohsin Ansari in
  their paper "Golden jackal optimization: A novel nature-inspired optimizer for
  engineering applications" (2022). 
  
  The link to the paper can be found here:

  https://www.sciencedirect.com/science/article/abs/pii/S095741742200358X

  ---------

  Attibutes of Class

  ---------

  self.f_obj : function we are trying to minimize
  self.dim: dimension of search space (default: 2)
  self.position_history_ : list of historical values of the male jackal
  self.fitness_history_ : list of historical values of the minimum fitness amongst the preys

  """

  def __init__(self,  f_obj, dim = 2):  
    self.position_history_ = []
    self.fitness_history_ = []
    self.f_obj_ = f_obj
    self.dim_ = dim


  def LF(self, beta):
    """
      Description: Levy Flight Function (used to update rl)
      
      Input: None
      Output: Random sample of the Levy Flight Function (float)
    """
    num = gamma(1 + beta)*np.sin(np.pi*beta/2)
    den = gamma((1 + beta)/2)*beta*(2**((beta-1)/2))
    sigm = (num/den)**(1/beta)
    mu, v = np.random.normal(0,1), np.random.normal(0,1) 
    return 0.01*(mu*sigm)/(v**(1/beta))

  
  def model(self, max_iter, size, beta=1.5, lb = -100, ub = 100):
    """
      Description: Function that runs the GJO algorithm. With each iteration, 
        we compute the function value of all potential points (preys) and find
        the two optimal ones (male & female jackal). We append the optimal point
        to position_history_ and its function value to fitness_history_.
        Then we find determine the energy level E and update the position of all
        potential points according to formula (4) and (5) (|E| > 1) or (12) and 
        (13) (|E| < 1).
    
      Input:
        max_iter (int): number of maximal iterations
        size (int): number of elements in the prey matrix
        beta (float) : beta parameter (default : 1.5)
        lb (float): lower bound for initialization of preys (default: -100)
        ub (float): upper bound for initialization of preys (default: 100)

      Output:
        None
    """

    # Initialize Prey matrix, t and energy level E0
    Prey = np.random.uniform(lb, ub, size = (size, self.dim_))
    t = 0
    E0 = [2*np.random.uniform(0,1)-1 for i in range(size)] 
    
    # Start main loop
    while t < max_iter :

      # 1/ Calculating the fitness of each prey
      FOA = [self.f_obj_(Prey[i]) for i in range(size)]
        
      # 2/ Update the position of the Male and the Female Jackal
      Y1 = Prey[FOA.index(min(FOA))].copy()
      self.position_history_.append(Y1)
      self.fitness_history_.append(min(FOA))
      FOA.pop(FOA.index(min(FOA)))
      Y2 = Prey[FOA.index(min(FOA))].copy()

      # 3/ Update the parameters E and rl and recalculate the position of each prey
      for i in range(size):                       
        E1 = 1.5*(1 - t/max_iter)
        E = E1*E0[i]
        rl = 0.05*self.LF(beta)
        if abs(E) > 1 :
          Y1_new = Y1 - E*abs(Y1 - rl*Prey[i])
          Y2_new = Y2 - E*abs(Y2 - rl*Prey[i])
          y = (Y1_new+Y2_new)/2
        else :
          Y1_new = Y1 - E*abs(rl*Y1 - Prey[i])
          Y2_new = Y2 - E*abs(rl*Y2 - Prey[i])
          y = (Y1_new+Y2_new)/2
        Prey[i] = y
      t+=1