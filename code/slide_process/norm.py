import numpy as np
from numpy import linalg as LA

def normalizeStaining(I, Io = 240, beta = 0.15, alpha = 1, HERef = np.asarray([0.5626, 0.2159, 0.7201, 0.8012, 0.4062, 0.5581]).reshape([3,2]), maxCRef = np.asarray([1.9705, 1.0308])):
    """

    Args:
        I: RGB input image.
        Io: transmitted light intensity.
        beta: OD threshold for transparent pixels.
        alpha: tolerance for the pseudo-min and pseudo-max.
        HERef: reference H&E OD matrix.
        maxCRef: reference maximum stain concentrations for H&E.

    Returns: 
        Inorm: normalized image.
        H : hematoxylin image.
        E : eosin image.
    """
    
    h, w = I.shape[:2]
    I = np.asarray(I, dtype = 'float32').reshape([h*w, 3],order='F')

    # calculate optical density
    OD = -np.log((I+1)/Io)

    # remove transparent pixels
    ODhat = OD[np.logical_not(np.any(OD < beta, 1)), :]

    # calculate eigenvectors and eigenvalues
    ev, V = LA.eig(np.cov(ODhat, rowvar = 0))

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    V =V[:,ev.argsort()]
    That = np.matmul(ODhat, V[:,-2:])

    # find the min and max vectors and project back to OD space
    phi = np.arctan2(That[:,1], That[:,0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)

    vMin = np.matmul(V[:,-2:], np.asarray([np.cos(minPhi), np.sin(minPhi)]))
    vMax = np.matmul(V[:,-2:], np.asarray([np.cos(maxPhi), np.sin(maxPhi)]))

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.asarray([vMin[:, np.newaxis], vMax[:, np.newaxis]])
    else:
        HE = np.asarray([vMax[:, np.newaxis], vMin[:, np.newaxis]])

    HE = np.squeeze(HE, axis = 2).transpose()
    # rows correspond to channels (RGB), columns to OD values
    Y = np.transpose(OD.reshape([h*w,3], order = 'F'))

    # determine concentrations of the individual stains
    C,_,_,_ = LA.lstsq(HE, Y, rcond=None)

    # normalize stain concentrations
    maxC = np.percentile(C, 99, axis = 1)
    C = (C.transpose() / maxC).transpose()
    C = (C.transpose() * maxCRef).transpose()

    # recreate the image using reference mixing matrix
    Inorm = Io*np.exp(np.matmul(-HERef, C))
    Inorm = Inorm.transpose().reshape([h, w, 3], order = 'F')
    print([np.min(Inorm), np.max(Inorm)])
    Inorm = (Inorm - np.min(Inorm))/(np.max(Inorm) - np.min(Inorm))*255
    print([np.min(Inorm), np.max(Inorm)])
    #Inorm[Inorm > 255] = 255
    #Inorm[Inorm < 0] = 0
    Inorm = np.asarray(Inorm, dtype = 'uint8')

    H = Io*np.exp(np.matmul(-HERef[:,0][:, np.newaxis], C[0,:][np.newaxis, :]))
    H = H.transpose().reshape([h, w, 3], order = 'F')
    H[H > 255] = 255
    H[H < 0] = 0
    H = np.asarray(H, dtype = 'uint8')

    E = Io*np.exp(np.matmul(-HERef[:,1][:, np.newaxis], C[1,:][np.newaxis, :]))
    E = E.transpose().reshape([h, w, 3], order = 'F')
    E[E > 255] = 255
    E[E < 0] = 0
    E = np.asarray(E, dtype = 'uint8')

    return Inorm, H, E
