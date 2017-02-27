import numpy as np
import scipy.interpolate
import pylab

def createBaseData (n):
    """ Given an integer n, returns n data points
    x and values y as a numpy.array."""
    xmax = 5.
    x = np. linspace (0, xmax , n)
    y =  x**2
    #make x -data somewhat irregular
    y += 1.5 * np. random.normal (size=len(x))
    return x, y


if __name__ == '__main__':
    n = 10
    x, y = createBaseData (n)
    # Develop a regular mesh for plot
    xfine = np.linspace(0.1 , 4.9, n * 100)
    # interpolate with piecewise linear func (p=1)
    y1 = scipy.interpolate.interp1d (x, y, kind='linear')
    # interpolate with quadratic fit(p=2)
    y2 = scipy.interpolate.interp1d (x, y, kind='quadratic')
    #Dynamic interpolation with splines of varying order
    y0 = scipy.interpolate.interp1d (x, y, kind='nearest')
    pylab.plot(x, y, 'o', color='yellow', label='ActualData Values')
    pylab.plot(xfine , y0(xfine), label='Nearest Fit')
    pylab.plot(xfine , y1(xfine), label='Linear Fit')
    pylab.plot(xfine , y2(xfine), label='Quadratic Fit')
    pylab. legend ()
    pylab. xlabel ('x values')
    pylab.show ()
