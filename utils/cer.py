import numpy as np
import math
import string

def cer(ref, pred):
    # Ref: true label
    # pred: predicted label

    # known alternative: WER
    # maybe combine?

    ref = ref.strip()
    pred = pred.strip()
    
    lp = len(pred)
    lr = len(ref)

    mat = [[0] * (lp+ 1) for _ in range(lr + 1)]

    for i,r in enumerate(ref,1):
        for j, p in enumerate(pred,1):
            if r == p:
                mat[i][j] = mat[i-1][j-1]
            else:
                s = mat[i-1][j-1]
                d = mat[i-1][j]
                i = mat[i][j-1]
                mat[i][j] = 1 + min(s,d,i)
    
    # ( S + D + I ) / N
    return 100.0 * mat[lr][lp] / lr
