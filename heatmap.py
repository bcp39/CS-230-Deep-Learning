from scipy.stats.stats import pearsonr 
import numpy

a = [1,4,6]
b = [1,2,3]   
print pearsonr(a,b)
print numpy.corrcoef(a,b)  