import numpy as np

class ERLS:

    def __init__(self):
        
        self.thetak = np.zeros((8,1))
        self.Ck = 1e4*np.identity(8)
        self.lambd = 0.98

        self.phik = np.zeros((6,1))
        self.uk = np.zeros((6,1))
        self.yk = np.zeros((2,1))
        self.ek = np.zeros((2,1))

    def estimate(self,u1,u2,y):

        self.phik = np.array([[self.uk[1,0]],
                              [self.uk[2,0]],
                              [self.uk[4,0]],
                              [self.uk[5,0]],
                              [-self.yk[0,0]],
                              [-self.yk[1,0]],
                              [self.ek[0,0]],
                              [self.ek[1,0]]], dtype=float)
        self.y_hat = np.matmul(self.phik.T,self.thetak)

        e_hat = y - self.y_hat
        Lk = np.matmul(self.Ck,self.phik)/(self.lambd + np.matmul(self.phik.T,np.matmul(self.Ck,self.phik)))
        self.thetak = self.thetak + (Lk*e_hat)
        self.Ck = (self.lambd**(-1))*np.matmul((np.identity(8) - np.matmul(Lk,self.phik.T)),self.Ck)

        y_hat = self.y_hat
        theta = self.thetak

        # Shift registers
        self.uk = np.array([[u1],
                            [self.uk[0,0]],
                            [self.uk[1,0]],
                            [u2],
                            [self.uk[3,0]],
                            [self.uk[4,0]]], dtype=float)
        self.yk = np.array([[y],
                            [self.yk[0,0]]], dtype=float)
        self.ek = np.array([[e_hat],
                            [self.ek[0,0]]], dtype=float)

        return y_hat, theta

    def resetShiftRegisters(self):

        self.phik = np.zeros((6,1))
        self.uk = np.zeros((6,1))
        self.yk = np.zeros((2,1))
        self.ek = np.zeros((2,1)) 

    def setTheta(self,theta):

        self.thetak = theta

    def modelPredict(self,u1,u2):

        self.phik = np.array([[self.uk[1,0]],
                              [self.uk[2,0]],
                              [self.uk[4,0]],
                              [self.uk[5,0]],
                              [-self.yk[0,0]],
                              [-self.yk[1,0]]], dtype=float)
        self.y_hat = np.matmul(self.phik.T,self.thetak)
        y_hat = self.y_hat

        # Shift registers
        self.uk = np.array([[u1],
                            [self.uk[0,0]],
                            [self.uk[1,0]],
                            [u2],
                            [self.uk[3,0]],
                            [self.uk[4,0]]], dtype=float)
        self.yk = np.array([[y_hat],
                            [self.yk[0,0]]], dtype=float)

        return y_hat
