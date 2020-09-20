import os
import sys
import random
import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    print("SimpleITK is required to this example!")
    sys.exit(-1)

# make sure ialm is found and imported
print(os.path.realpath(__file__))
sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__)),'..')
import ialm

def main(argv=None):
    if argv is None:	
        argv = sys.argv

    if len(argv) != 5:
        print("Usage : python %s <ChekerboardImage> <OutlierFraction> <CorruptedImage> <LowrankImage>" %sys.argv[0])
        sys.exit(1)
    
    # outlier fraction
    p = float(sys.argv[2])

    # read Image
    I = sitk.ReadImage(argv[1])

    # data for processing
    X = stik.GetArrayFromImage(I)

    # number of pixel
    N = np.prod(X.shape)

    eps = np.round(np.round.uniform(-10, 10, 100))
    idx = np.random.random_integers(0, N-1, np.round(N*p))
    X.ravel()[idx] = np.array(200+eps, dtype=np.unit8)

    # write outlier image
    J = sitk.GetImageFromArray(X)
    sitk.WriteImage(J, sys.argv[3])

    # decompse X into L+S
    L, S, _ = ialm.recover(X)

    C = sitk.GetImageFromArray(np.asarray(L, dtype=np.uint8))
    sitk.WriteImage(C, sys.argv[4])

    # Compute mean-square error and Frobenius norm
    print("MSE: %.4g" % np.sqrt(np.asmatrix((L-sitk.GetArrayFromImage(I))**2).sum()))
    print("Frobeniue-Norm: %.4g" %np.linalg.norm(L-sitk.GetArrayFromImage(I), ord='fro'))

if __name__ == '__main__':
    main()
