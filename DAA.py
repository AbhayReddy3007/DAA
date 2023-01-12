A = ""
B = ""
m, n = 0, 0
import string
import random

def max(a, b, c):
    if a > b and a > c:
        return a
    elif b > a and b > c:
        return b
    else:
        return c

def nmov(M,x,y):
    diag = M[x-1][y-1]
    up = M[x-1][y]
    left = M[x][y-1]
    if diag >= up and diag >= left:
        return 1 if diag != 0 else 0
    elif up > diag and up >= left:
        return 2 if up != 0 else 0
    elif left > diag and left > up:
        return 3 if left != 0 else 0
    else:
        print("okay")
        exit(1)

alignA = ""
alignB = ""

def traceback(M,x,y):
    diag, up, left = 1, 2, 3
    move = nmov(M,x,y)
    global alignA
    global alignB
    if move == 1:
        print(M[x][y])
        alignA += A[x-1]
        alignB += B[y-1]
        traceback(M,x-1,y-1)
    elif move == 2:
        print(M[x][y])
        alignA += A[x-1]
        alignB += '_'
        traceback(M,x-1,y)
    elif move==3:
        print(M[x][y])
        alignA += '_'
        alignB += B[y-1]
        traceback(M,x,y-1)
    if(move==0):
        alignA += B[x-1]
        alignB += A[y-1]
        alignA = alignA[::-1]
        alignB = alignB[::-1]
        print(alignA)
        print(alignB) 


def ScoringMatrix():
    match = 5
    mismatch = -4
    c = -1  # gap
    import numpy as np
    max_score=-999
    max_pos_i, max_pos_j = 0, 0
    m, n = len(A), len(B)
    M = [[0 for j in range(m + 1)] for i in range(n + 1)]
    
    for i in range(m + 1):
        M[i][0] = 0
    for j in range(n + 1):
        M[0][j] = 0

    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if A[i - 1] == B[j - 1]:
                M[i][j] = M[i - 1][j - 1] + match
            else:
                M[i][j] = max(M[i - 1][j - 1] + mismatch,
                              M[i - 1][j] + mismatch,
                              M[i][j - 1] + mismatch)
            if M[i][j] > max_score:
                max_score = M[i][j]
                max_pos_i, max_pos_j = i, j
                
    print("     ", end = "")
    for i in range(m):
        print(B[i],end="  ")
    print("\n  ", end = "")
    for i in range(n+1):
        if i > 0:
            print(A[i - 1], end = " ")
        for j in range(m+1):
            print(M[i][j], end = "  ")
        print()
    print("Traceback sequence: ")
    traceback(M, max_pos_i, max_pos_j)

def randStr(chars = string.ascii_uppercase + string.digits, N=16):
	return ''.join(random.choice(chars) for _ in range(N))

if __name__ == "__main__":
    print("String A: ")
    A = randStr(chars='abcd')
    print(A)
    print("String B: ")
    B = randStr(chars='abcd')
    print(B)
  
    ScoringMatrix()
