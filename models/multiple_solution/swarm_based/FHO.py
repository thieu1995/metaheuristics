# -*- coding: utf-8 -*-
"""FHO - Fire Hawk Optimizer Algorithm

Algorithm based on the publication : https://link.springer.com/article/10.1007/s10462-022-10173-w
Original Authors: Mahdi Azizi, Siamak Talatahari & Amir H. Gandomi
Original Python file developped by : Adel Remadi and Hamza Lamsaoub  (Students at CentraleSupélec)
"""

import numpy as np
import matplotlib.pyplot as plt
from models.multiple_solution.root_multiple import RootAlgo


class FHO(RootAlgo):
    """ Class for FHO 
    ---------
    self.cost_function: fucntion to optimize
    self.candidates : List of Candidates
    self.d: dimension of the problem
    self.Fhawks : list of fire hawks
    self.preys : list of preys
    self.territories : territory of each hawk
    self.N : Number of solution candidates (int)
    self.minD : numpy array of dimension (d,) containing the MIN value for each variable
    self.maxD : numpy array of dimension (d,) containing the MAX value for each variable
    self.N : Number of solution candidates (int)
    self.max_generations: Maximum number of generated candidates (int)
    self.best_costs: list of best costs
    """
    
    def __init__(self, minD , maxD, N, function , max_generations=200):
        self.max_generations = max_generations
        self.minD            = minD
        self.maxD            = maxD
        self.cost_function   = function
        self.N               = N
        self.d               = len(minD) #Number of decision variables
        self.Xcandidates     = np.random.uniform(minD,maxD,(N,self.d)) #Initial Solution candidates (Nxd)
        self.best_costs      = []
        self.costs_iter      = []
        self.minimal_p       = None
        
        
    def territories(self,Fhawks,Preys):
        #Computing territories using the euclidien distance
        preys_left=Preys.copy()
        territories={i:np.array([]) for i in range(len(Fhawks))}
        for i in range(len(Fhawks)):
            #distance with respect to Fire hawk i           
            D=np.linalg.norm(Fhawks[i]-preys_left,axis=1)
            
            #Get territory of fire Hawk i 
            sorted_preys_idx=np.argsort(D)
            alpha=np.random.randint(1,len(preys_left)-1) if len(preys_left)-1>1 else 1
            my_preys=sorted_preys_idx[:alpha]
            territories[i]=preys_left[my_preys]
            preys_left=preys_left[sorted_preys_idx[alpha:]]          
            if len(preys_left)==0:
                break
        if len(preys_left)>0:
            territories[len(Fhawks)-1]=np.array(list(territories[len(Fhawks)-1])+list(preys_left))
        return territories
            
                      
    def minimize_FHO(self):
        ## Fire hawk algorithm to minimize the cost function       
        
        d=self.d
        Xcandidates=self.Xcandidates
        N=self.N
        max_generations = self.max_generations  
        minD=self.minD
        maxD=self.maxD
        Costfunction = self.cost_function
        
        #Evaluate the cost function for all candidate vectors
        Cost= np.array([Costfunction(Xcandidates[i]) for i in range(N)])
        
        #Randomly set a number of Hawks between 1 and 20% of N
        num_Hawks = np.random.randint(1,int(N/5)+1) if 1<int(N/5)+1 else 1
        
        #Ordering candidates
        Xcandidates = Xcandidates[np.argsort(Cost)]
        Cost.sort()
        SP=Xcandidates.mean(axis=0)
         
        
        #Select fire hawks
        Fhawks= Xcandidates[:num_Hawks]
        
        #Select the Preys dim(N-num_Hawks,d)
        Preys = Xcandidates[num_Hawks:]
        
        #get territories
        territories=self.territories(Fhawks,Preys)
        
        #update best
        GB=Cost[0]
        Best_Hawk=Xcandidates[0]

        #Counter
        FEs=N

        ## Main Loop
        while FEs < max_generations:
            PopTot=[]
            Cost=[]
            #Movement of Fire Hawk for all territories
            for i in territories:
                PR=territories[i].copy()
                FHl=Fhawks[i].copy()
                SPl=PR.mean(axis=0) if len(territories[i]) > 0 else np.zeros(FHl.shape)
                a,b=np.random.uniform(0,1,size=2)
                FHnear  =Fhawks[np.random.randint(num_Hawks)]                 
                FHl_new =FHl+(a*GB-b*FHnear)
                FHl_new = np.maximum(FHl_new,minD)
                FHl_new = np.minimum(FHl_new,maxD)
                PopTot.append(list(FHl_new))
                
                #Movement of the preys following Fire Hawks movement
                for q in range(len(PR)): 
                    a,b=np.random.uniform(0,1,size=2)
                    PRq_new1=PR[q].copy()+((a*FHl-b*SPl))
                    PRq_new1= np.maximum(PRq_new1,minD)
                    PRq_new1 = np.minimum(PRq_new1,maxD)
                    PopTot.append(list(PRq_new1))
                    
                    #Movement of the preys outside of territory
                    a,b      =np.random.uniform(0,1,size=2)
                    FHAlter  =Fhawks[np.random.randint(num_Hawks)] 
                    PRq_new2 =PR[q].copy()+((a*FHAlter-b*SP));
                    PRq_new2 = np.maximum(PRq_new2,minD)
                    #The following line for PRq_new2 differs from original algorithm in matlab code (max instead of min):
                    # Effects observed through our testing:
                    # 1/ It converges faster and the costs of the subsequent iterations will tend to decrease (less chaotic behavior than with np.minimun)
                    # 2/ In higher dimensions, it converge to the right solution! (while with np.minimum it does not)
                    PRq_new2 = np.maximum(PRq_new2,maxD)
                    PopTot.append(list(PRq_new2))
                             
            #Get cost
            PopTot=np.array(PopTot)
            for i in range(len(PopTot)):
                Cost.append(Costfunction(PopTot[i]))
                FEs = FEs+1
     
            #Create a new population of Hawks and Preys
            order_idx=np.argsort(Cost)
            Cost.sort()
            PopTot=np.array(PopTot)[order_idx]
            num_Hawks = np.random.randint(1,int(N/5)+1) if 1<int(N/5)+1 else 1
            BestPop=PopTot[0]
            SP=PopTot.mean(axis=0)
            Fhawks=PopTot[:num_Hawks]
            Preys=PopTot[num_Hawks:]          
               
            #Get new territories
            territories=self.territories(Fhawks,Preys)
           
            # Update Global Best Cost (if relevant) 
            if Cost[0]<GB:
                BestPos=BestPop
                GB=Cost[0]
                self.best_costs.append(GB)
                self.minimal_p=Fhawks[0]
            else:
                self.best_costs.append(GB)
           
            #Track the iteration calculated cost
            self.costs_iter.append(Cost[0])

        #Return Global Best and argmin
        return (GB,self.minimal_p)
            
    def plot_costs(self):
        #Plot cost evolution
        vals=self.costs_iter
        n=len(vals)
        vals2=self.best_costs
        plt.figure()
        plt.title("Cost per iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.plot(np.arange(n),vals,label="Iteration Cost")
        plt.plot(np.arange(n),vals2, label="Global Best Cost")
        plt.xticks(np.arange(n))
        plt.legend()
        plt.show()