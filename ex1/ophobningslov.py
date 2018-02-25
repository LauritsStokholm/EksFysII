import numpy as np

def konf(X, Y, sds):
    # X = np.array(x1, x2, x3, ... xn)
    # hvor xi = np.array([1, 2, 3, 4, ... n]) (uafhængig / her vinkel)
    
    # Y = np.array([1, 2, 3, ... n])
    
    # sds = np.array([a, b, c, ... n]) # selvvalgte n tal

    diff_X = np.zeros(np.size(X)/np.size(Y))

    for i in range(0, np.size(diff_X)):
        diff_X[i] = np.diff(Y) / np.diff(X[i])

    errr = diff_X *sds
    sigma = np.sqrt(sum(errr**2))
    sigma_array = np.ones(np.size(Y)) * sigma

    konf = np.array([Y-sigma_array], [Y+sigma_array])

    return(konf)


