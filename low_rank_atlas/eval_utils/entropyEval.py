import numpy
import SimpleITK as sitk


def __range(a, bins):
    '''Compute the histogram range of the values in the array a according to
    scipy.stats.histogram.'''
    a = numpy.asarray(a)
    a_max = a.max()
    a_min = a.min()
    s = 0.5 * (a_max - a_min) / float(bins - 1)
    return (a_min - s, a_max + s)
 

def __entropy(data):
    '''Compute entropy of the flattened data set (e.g. a density distribution).'''
    # normalize and convert to float
    data = data/float(numpy.sum(data))
    # for each grey-value g with a probability p(g) = 0, the entropy is defined as 0, therefore we remove these values and also flatten the histogram
    data = data[numpy.nonzero(data)]
    # compute entropy
    return -1. * numpy.sum(data * numpy.log2(data))



def imageEntropy(image_fn,num_bins=100):
    im = sitk.ReadImage(image_fn)
    im_array= sitk.GetArrayFromImage(im) # get numpy array
    range_1= __range(im_array, num_bins)
    hist, _ = numpy.histogram(im_array, bins=num_bins, range=range_1)
    entropy = __entropy(hist)
    return entropy



uatlas = "/Users/xiaoxiaoliu/work/data/BRATS/BRATS-2/Image_Data/results/UAB_ANTS_16/L0_Iter5_atlas.nrrd"
latlas = "/Users/xiaoxiaoliu/work/data/BRATS/BRATS-2/Image_Data/results/LAB_ANTS_16/L0_Iter5_atlas.nrrd"
#uatlas = "/Users/xiaoxiaoliu/work/data/BRATS/BRATS-2/Image_Data/results/UAB_ANTS_16/atlas_L0_Iter5.png"
#latlas = "/Users/xiaoxiaoliu/work/data/BRATS/BRATS-2/Image_Data/results/LAB_ANTS_16/atlas_L0_Iter5.png"
print "low-rank atlas entroy:", imageEntropy(latlas)
print "unbiased atlas entropy:", imageEntropy(uatlas)

