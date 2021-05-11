import numpy as np

class dynamics:
    """
    This dynamics class is a stateful simple 2D double integrator (2 pos dimentions, 2 velocity)
    step: updates the internal state according to commanded action
    reset: samples new initial state
    """
    
    def __init__(self, stochastic=True):
        self.A = np.eye(4)
        self.dt = 0.5
        self.A[0,2] = self.dt
        self.A[1,3] = self.dt
        self.A[0,3] = 0.5*self.dt
        
        self.B = np.zeros((4,2))
        self.B[2,0] = 0.5
        self.B[3,1] = 1.5
        
        self.stochastic = stochastic
        self.cov = np.array([0.05, 0.05, 0.025, 0.025])
        
        self.reset()
    
    def step(self,u):
        xp = self.A @ self.x + self.B @ u
        if self.stochastic:
            xp += self.cov * np.random.randn(4)
        
        self.x = xp
        return xp
    
    def get_state(self):
        return self.x
    
    def reset(self):
        self.x = np.random.randn(4)
        self.x[0:2] *= 5
        return self.x

class cost:
    
    def __init__(self):
        self.Q = np.eye(4)
        self.Q[2,2] *= 0.1
        self.Q[3,3] *= 0.1

        self.R = 0.01*np.eye(2)
    
    def evaluate(self,x,u):
        return x @ self.Q @ x + u @ self.R @ u
        