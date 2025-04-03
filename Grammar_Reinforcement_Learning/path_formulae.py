
import torch
# import numpy.linalg as lin
import time


# 2_ 6 Path new formulae at edge level :
def path1(A):
    return A
def path2(A):
    A2 = A@A
    I = torch.eye(A.shape[0])
    J = torch.ones(A.shape) - I
    return A2*J


def path3(A):
    A2 = A@A
    I = torch.eye(A.shape[0])
    J = torch.ones(A.shape) - I
    AJ =A.sum(1).unsqueeze(1)-A
    A2J = A2*J
    return A@A2J*J - AJ*A

def path4(A):
    A2 = A@A
    I = torch.eye(A.shape[0])
    J = torch.ones(A.shape) - I
    AJ =A.sum(1).unsqueeze(1)-A
    A2J = A2*J
    AJA = AJ*A
    A2A = A2*A
    A2AJ = A2A.sum(1).unsqueeze(1) - A2A
    C3f = A@A2J*J - AJA
    return A@C3f*J - AJA@A*J - A2AJ*A + 2*A2A



def path5(A):
    A2 = A@A
    I = torch.eye(A.shape[0])
    J = torch.ones(A.shape) - I
    AJ =A.sum(1).unsqueeze(1)-A
    A2J = A2*J
    AJA = AJ*A
    A2A = A2*A
    A2AJ = A2A.sum(1).unsqueeze(1) - A2A
    C3f = A@A2J*J - AJA
    C4f = A@C3f*J - AJA@A*J - A2AJ*A + 2*A2A
    C3fJ = A*(A*C3f).sum(1).unsqueeze(1) - A*C3f
    return J*(A@C4f) - (A2AJ)*A2J - C3fJ - (AJ)*C3f + C3f + A*C3f - 4* A2A + 2* A*A2*A2 + 3*A2A@A*J



def path6(A):
    A2 = A@A
    I = torch.eye(A.shape[0])
    J = torch.ones(A.shape) - I
    AJ =A.sum(1).unsqueeze(1)-A
    A2J = A2*J
    AJA = AJ*A
    A2A = A2*A
    AA2A = A@A2A
    A2AA = A2A@A
    A2AJ = A2A.sum(1).unsqueeze(1) - A2A
    P3f = A@A2J*J - AJA
    P4f = A@P3f*J - AJA@A*J - A2AJ*A + 2*A2A
    P3fJ = (A*P3f).sum(1).unsqueeze(1) - A*P3f
    P5f = J*(A@P4f) - (A2AJ)*A2J -A* P3fJ - (AJ)*P3f + P3f + A*P3f - 4* A2A + 2* A*A2*A2 + 3*A2A@A*J
    P4fJ = (A*P4f).sum(1).unsqueeze(1) - A*P4f
    return (J*(A@P5f) - (A2AJ)*P3f - A*P4fJ - AJ*P4f + P4f + A*P4f +3*(A*P3f)@A*J 
            - A2J*P3fJ  + 4*A2*A*P3f + 3*A2A@A2J*J + A2*A2*A2*J + 3*A2A*A2@A*J
            -4*A2*A2*J - 8*A*AA2A - 8*A*A2AA - 4*A2AA*J - 3*A*A2AJ + 17*A2A + 3*A2J)

def path(A,n):
    f = [path1,path2,path3,path4,path5,path6]
    return f[n-1](A)


def cycle(A,n):
    return A*path(A,n-1)





