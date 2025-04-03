import torch


# 2_ 5 Path at edge level :
def path2(A):
    A2 = A@A
    I = torch.eye(A.shape[0])
    J = torch.ones(A.shape) - I
    return A2*J

def path3(A):
    A2 = A@A
    A3 = A2@A
    I = torch.eye(A.shape[0])
    J = torch.ones(A.shape) - I
    A2I =A2*I
    return A3*J - A@A2I - A2I@A + A 

def path3f(A):
    A2 = A@A
    I = torch.eye(A.shape[0])
    J = torch.ones(A.shape) - I
    A2I =A2*I
    A2J = A2*J
    return A@A2J*J - A2I@A + A 

def path3ff(A):
    A2 = A@A
    I = torch.eye(A.shape[0])
    J = torch.ones(A.shape) - I
    AJ =A.sum(1).unsqueeze(1)-A
    A2J = A2*J
    return A@A2J*J - AJ*A

def path4(A):
    A2 = A@A
    A3 = A2@A
    A4 = A3@A
    I = torch.eye(A.shape[0])
    J = torch.ones(A.shape) - I
    A2I = A2*I
    A2J = A2*J
    A3I = A3*I
    return (A4-A@(A2I)@A)*J + 2*A2J - A2J@A2I - A2I@A2J - A@A3I - A3I@A + 3*A*A2



def path4f(A):
    A2 = A@A
    I = torch.eye(A.shape[0])
    J = torch.ones(A.shape) - I
    AJ =A.sum(1).unsqueeze(1)-A
    A2J = A2*J
    AJA = AJ*A
    A2A = A2*A
    A2AJ = A2A.sum(1).unsqueeze(1) - A2A
    return A@(A@A2J*J)*J - A@(AJA)*J - AJA@A*J - A2AJ*A + 2*A2A

def path4ff(A):
    A2 = A@A
    I = torch.eye(A.shape[0])
    J = torch.ones(A.shape) - I
    AJ =A.sum(1).unsqueeze(1)-A
    A2J = A2*J
    AJA = AJ*A
    A2A = A2*A
    A2AJ = A2A.sum(1).unsqueeze(1) - A2A
    # print(A@(AJA.T)*J + AJA@A*J + A@A2A + A2A@A)
    return    A@(AJA)*J + AJA@A*J 

def path5(A):
    A2 = A@A
    A3 = A2@A
    A4 = A3@A
    A5 = A4@A
    I = torch.eye(A.shape[0])
    J = torch.ones(A.shape) - I
    A2I = A2*I
    A2ImI = A2I - I
    A2J = A2*J
    AA2 = A*A2
    A3I = A3*I
    P3 = A3*J - A@A2I - A2I@A + A
    AP3 = A*P3
    AP31 = torch.diag((AP3).sum(1))
    return (A5-A@(A2I)@(A2ImI) - A2I@A@A2I - (A2ImI)@A2I@A - A@A2ImI@A2J - A2J@A2ImI@A
            - A2I@P3 - P3@A2ImI - A3I@A2J - A2J@A3I - A@A3I@A - AA2
            + 3*(A@AA2 + AA2@A) - A@AP31 - AP31@A + 3*AP3 + 3*AA2*(A2-(A2>0)*1.))*J 

def path(A,n):
    f = [path2,path3ff,path4f,path5]
    return f[n-2](A)


def cycle(A,n):
    return A*path(A,n-1)


