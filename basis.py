# from create_constant_basis import create_constant_basis
import numpy as np

from fRegress import *
from fd import *
from smooth_basis import *
from linmod import *
from fourier import *

#########
# Important!!!
# Currently, only support constant and bspline basis
#########
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
#  This function creates a bspline functional data basis.
#  Arguments
#  RANGEVAL...an array of length 2 containing the lower and upper
#             boundaries for the rangeval of argument values,
#             or a positive number, in which case command
#             rangeval <- c(0, rangeval) is executed.
#             the default is c(0,1)
#  NBASIS  ...the number of basis functions.  This argument must be
#             supplied, and must be a positive integer.
#  NORDER  ...order of b-splines (one higher than their degree).  The
#             default of 4 gives cubic splines.
#  BREAKS  ...also called knots, these are a non-decreasing sequence
#             of junction points between piecewise polynomial segments.
#             They must satisfy BREAKS[1] = RANGEVAL[1] and
#             BREAKS[NBREAKS] = RANGEVAL[2], where NBREAKS is the total
#             number of BREAKS.  There must be at least 2 BREAKS.
#  There is a potential for inconsistency among arguments NBASIS, NORDER,
#  and BREAKS since
#             NBASIS = NORDER + LENGTH(BREAKS) - 2
#  An error message is issued if this is the case.  Although previous
#  versions of this function attempted to resolve this inconsistency in
#  various ways, this is now considered to be too risky.
#  DROPIND ...A vector of integers specifiying the basis functions to
#             be dropped, if any.  For example, if it is required that
#             a function be zero at the left boundary, this is achieved
#             by dropping the first basis function, the only one that
#             is nonzero at that point.
#  QUADVALS...A NQUAD by 2 matrix.  The firs t column contains quadrature
#             points to be used in a fixed point quadrature.  The second
#             contains quadrature weights.  For example, for (Simpson"s
#             rule for (NQUAD = 7, the points are equally spaced and the
#             weights are delta.*[1, 4, 2, 4, 2, 4, 1]/3.  DELTA is the
#             spacing between quadrature points.  The default is
#             matrix("numeric",0,0).
#  VALUES ... A list, with entries containing the values of
#             the basis function derivatives starting with 0 and
#             going up to the highest derivative needed.  The values
#             correspond to quadrature points in QUADVALS and it is
#             up to the user to decide whether or not to multiply
#             the derivative values by the square roots of the
#             quadrature weights so as to make numerical integration
#             a simple matrix multiplication.
#             Values are checked against QUADVALS to ensure the correct
#             number of rows, and against NBASIS to ensure the correct
#             number of columns.
#             The default value of is VALUES is vector("list",0).
#             VALUES contains values of basis functions and derivatives at
#             quadrature points weighted by square root of quadrature weights.
#             These values are only generated as required, and only if slot
#             QUADVALS is not matrix("numeric",0,0).
#  BASISVALUES...A vector of lists, allocated by code such as
#             vector("list",1).
#             This field is designed to avoid evaluation of a
#             basis system repeatedly at a set of argument values.
#             Each list within the vector corresponds to a specific set
#             of argument values, and must have at least two components,
#             which may be tagged as you wish.
#             The first component in an element of the list vector contains the
#             argument values.
#             The second component in an element of the list vector
#             contains a matrix of values of the basis functions evaluated
#             at the arguments in the first component.
#             The third and subsequent components, if present, contain
#             matrices of values their derivatives up to a maximum
#             derivative order.
#             Whenever function getbasismatrix is called, it checks
#             the first list in each row to see, first, if the number of
#             argument values corresponds to the size of the first dimension,
#             and if this test succeeds, checks that all of the argument
#             values match.  This takes time, of course, but is much
#             faster than re-evaluation of the basis system.  Even this
#             time can be avoided by direct retrieval of the desired
#             array.
#             For example, you might set up a vector of argument values
#             called "evalargs" along with a matrix of basis function
#             values for these argument values called "basismat".
#             You might want too use tags like "args" and "values",
#             respectively for these.  You would then assign them
#             to BASISVALUES with code such as
#               basisobj$basisvalues <- vector("list",1)
#               basisobj$basisvalues[[1]] <-
#                               list(args=evalargs, values=basismat)
#  BASISFNNAMES ... Either a character vector of length NABASIS
#             or a single character string to which NORDER, "." and
#             1:NBASIS are appended by the command
#                paste(names, norder, ".", 1:nbreaks, sep="").
#             For example, if norder = 4, this defaults to
#                     'bspl4.1', 'bspl4.2', ... .
#  Returns
#  BASISFD ...a functional data basis object

#  Last modified  11 February 2015 by Jim Ramsay

#  -------------------------------------------------------------------------
#  Default basis for missing arguments:  A B-spline basis over [0,1] of
#    of specified norder with norder basis functions.
#    norder = 1 = one basis function = constant 1
#    norder = 2 = two basis functions = 2 right triangles,
#      one left, the other right.  They are a basis for straight lines
#      over the unit interval, and are equivalent to a monomial basis
#      with two basis functions.  This B-spline system can be
#      explicitly created with the command
#                create.bspline.basis(c(0,1), 2, 2)
#    norder = 3 = three basis functions:  x^2, x-(x-.5)^2, (x-1)^2
#    norder = 4 = default = 4 basis functions
#      = the simplest cubic spline basis
#  -------------------------------------------------------------------------

def create_bspline_basis(rangeval=[], nbasis=4, norder=4, breaks=[], dropind=[], quadvals=[], values=[],
                         basisvalues=[], names='bspline'):
    type = names

    # check RANGEVAL
    # 1.1 first check breaks is either None or is numeric with positive length

    if breaks is not None and len(breaks) != 0:
        if min(np.diff(breaks)) < 0:
            raise ValueError("One or more breaks differences are negative.")
        if len(breaks) < 1:
            breaks = []
        if sum(np.isnan(breaks)) > 0:
            raise ValueError("breaks contains NAs; not allowed")
        if sum(np.isinf(breaks)) > 0:
            raise ValueError("breaks contains Infs; not allowed")

    if len(rangeval) < 1:
        if len(breaks) == 0:
            rangeval = [0, 1]
        else:
            rangeval = [min(breaks), max(breaks)]
            if np.diff(rangeval) == 0:
                raise ValueError("diff(range(breaks)) == 0, not allowed.")
    else:
        nNa = sum(np.isnan(rangeval))
        if nNa > 0:
            raise ValueError("@param rangeval contains NA, not allowed.")

    if len(rangeval) == 1:
        if rangeval <= 0:
            raise ValueError("'rangeval' a single value is not positive, is", rangeval)
        rangeval = np.array([0, rangeval])

    if len(rangeval) > 2:
        if len(breaks) != 0:
            raise ValueError("breaks can not be provided with len(rangeval) > 2")
        breaks = rangeval
        rangeval = np.array([min(breaks), max(breaks)])

    if rangeval[0] >= rangeval[1]:
        raise ValueError("rangeval[0] must be less than rangeval[1].")

    # 2. check norder

    if norder <= 0 or norder % 1 > 0:
        raise ValueError("norder must be a single positive integer.")

    # 3. check nbasis

    nbreaks = len(breaks)
    if nbasis is not None:
        if nbasis <= 0 or nbasis % 1 > 0:
            raise ValueError("nbasis must be a single positive integer.")
        elif nbasis < norder:
            raise ValueError("nbasis must be at least norder")
        # 4. check breaks
        if len(breaks) != 0:
            if nbreaks < 2:
                raise ValueError("Number of values in argument 'breaks' less than 2.")
            if breaks[0] != rangeval[0] or breaks[nbreaks - 1] != rangeval[1]:
                raise ValueError("Range if argument 'breaks' not identical to that of argument 'rangeval'.")
            if min(np.diff(breaks)) < 0:
                raise ValueError("values in argument 'breaks' are decreasing.")
            if nbasis != norder + nbreaks - 2:
                raise ValueError("Relation nbasis = norder + length(breaks) -2 does not hold.")
        else:
            breaks = np.linspace(rangeval[0], rangeval[1], nbasis - norder + 2)
            nbreaks = len(breaks)
    else:
        if len(breaks) == 0:
            nbasis = norder
        else:
            nbasis = len(breaks) + norder - 2

    # 5. Set up the PARAMS vector, which contains only the interior knots
    if nbreaks > 2:
        params = breaks[1: (nbreaks - 1)]
    else:
        params = []

    # 6. set up basis object

    basisobj = basis(type=type, rangeval=rangeval, nbasis=nbasis, params=params, dropind=dropind, quadvals=quadvals,
                     values=values,
                     basisvalues=basisvalues)

    # 7. names

    return basisobj


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
def transpose(matrix):
    transposed = []
    for i in range(len(matrix[0])):
        transposed.append([row[i] for row in matrix])
    return transposed


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
def create_constant_basis(rangeval=[0, 1], names='const', axes=None):
    if len(rangeval) == 1:
        if rangeval[0] <= 0:
            raise ValueError("RANGEVAL a single value that is not positive. ")
        rangeval = [0, rangeval[0]]

    type = names
    nbasis = 1
    params = [0]
    dropind = [0]
    quadvals = [0]
    values = []
    basisvalues = []

    basisobj = basis(type=type, rangeval=rangeval, nbasis=nbasis, params=params, dropind=dropind, quadvals=quadvals,
                     values=values, basisvalues=basisvalues)
    return basisobj


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
class basis:

    def __init__(self, type='bspline', rangeval=[0, 1], nbasis=2, params=[], dropind=[], quadvals=[], values=[],
                 basisvalues=[], names=None):

        # check type
        if type == 'bspline' or type == 'Bspline' or type == 'spline' or type == 'Bsp' or type == 'bsp':
            self.type = 'bspline'
        elif type == 'con' or type == 'const' or type == 'constant':
            self.type = 'const'
        elif type == 'exp' or type == 'expon' or type == 'exponential':
            self.type = 'expon'
        elif type == 'Fourier' or type == 'fourier' or type == 'fou' or type == 'Fou':
            self.type = 'fourier'
        else:
            raise ValueError(
                "@param type error, only b-spline, exponential and constant basis functions are supported now.")

        # check rangeval
        # rangeval should be a numpy array of length 2 containing the lower and upper boundaries

        if len(rangeval) != 2:
            raise ValueError("rangeval should be a numpy array of length 2 containing the lower and upper boundaries")
        elif rangeval[1] <= rangeval[0]:
            raise ValueError("Argument rangeval is not strictly increasing.")
        else:
            self.rangeval = rangeval

        # check nbasis

        if nbasis <= 0:
            raise ValueError("Argument nbasis is not positive.")
        elif round(nbasis) != nbasis:
            raise ValueError("Argument basis is not an integer.")
        else:
            self.nbasis = nbasis

        # check quadvals

        if len(quadvals) != 0 and quadvals is not None:
            dim = np.shape(quadvals)
            self.nquad = dim[0]
            self.ncol = 1
            # if self.nquad == 2 and self.ncol > 2:
            #     quadvals = transpose(quadvals)
            #     self.nquad = quadvals.shape[0]
            #     ncol = quadvals.shape[1]
            # if self.nquad < 2:
            #     raise ValueError("Less than two quadrature points are supplied.")
            # if ncol != 2:
            #     raise ValueError("'quadvals' does not have two columns.")

        # check VALUES is present, and set to a single empty list if not.

        if len(values) != 0 and values is not None:
            if values[0] != self.nquad:
                raise ValueError("Number of rows in 'values' not equal to number of quadrature points")
            if values[1] != self.nbasis:
                raise ValueError("Number of columns in 'values' not equal to number of basis functions")
        else:
            values = []

        # check BASISVALUES is present, and set to list() if not
        # If present, it must be a two-dimensional list created by a command like
        # listobj = np.array([2,3])

        if len(basisvalues) != 0 and basisvalues is not None:
            sizeves = np.shape(basisvalues)
            if len(sizeves) != 2:
                raise ValueError("BASISVALUES is not 2-dimensional.")
            # Waiting to check
            # for (i in 1:sizevec[1]) {
            # if (length(basisvalues[[i, 1]]) != dim(basisvalues[[i, 2]])[1]) stop(
            # paste("Number of argument values not equal number",
            # "of values."))
            # }
        else:
            basisvalues = None

        self.basisvalues = basisvalues

        # check if DROPIND is presentm and set to default if not

        if len(dropind) > 0:
            ndrop = len(dropind)
            if ndrop > self.nbasis:
                raise ValueError("Too many index values in DROPIND.")
            dropind.sort()

        self.dropind = dropind
        # Waiting to check
        # if (ndrop > 1 & & any(diff(dropind)) == 0)
        #     stop('Multiple index values in DROPIND.')
        # for (i in 1:ndrop) {
        # if (dropind[i] < 1 | | dropind[i] > nbasis)
        # stop('A DROPIND index value is out of range.')
        # }

        # check values
        # nvalues = length(values)
        # if (nvalues > 0 & & length(values[[1]] > 0)) {
        # for (ivalue in 1:nvalues) {
        #     derivvals = values[[ivalue]]
        # derivvals = derivvals[, -dropind]
        # values[[ivalue]] = derivvals
        # }
        # }
        # }
        self.values = values

        # select the appropriate type and process

        if self.type == 'const':
            self.params = 0
        elif self.type == 'bspline':
            if params is not None:
                nparams = len(params)
                if nparams > 0:
                    if params[0] < self.rangeval[0]:
                        raise ValueError("Smallest value in BREAKS not within RANGEVAL")
                    if params[nparams - 1] >= self.rangeval[1]:
                        raise ValueError("Largest value in BREAKS not within RANGEVAL")

        self.params = params

        self.names = names

    def summary(self):

        print("\n Type: ", self.type, "\n")
        print("\n Range: ", self.rangeval[1], "to ", self.rangeval[2], "\n")
        if self.type != 'const':
            print("\n Number of basis functions: ", self.nbasis, "\n")
        if len(self.dropind) > 0:
            print(len(self.dropind), "indices of basis functions to be dropped.")

    def __eq__(self, other):

        type1 = self.type
        range1 = self.rangeval
        nbasis1 = self.nbasis
        pars1 = self.params
        drop1 = self.dropind

        type2 = other.type
        range2 = other.rangeval
        nbasis2 = other.nbasis
        pars2 = other.params
        drop2 = other.dropind

        if type1 != type2:
            return False

        if range1[0] != range2[0] or range1[1] != range2[1]:
            return False

        if nbasis1 != nbasis2:
            return False

        if np.all(drop1 != drop2) != 0:
            return False

        return True

    def __mul__(self, other):
        # Important!!!
        # Currently, this method only support constant and bspline basis
        range1 = self.rangeval
        range2 = other.rangeval

        if range1[0] != range2[0] or range1[1] != range2[1]:
            raise ValueError("Ranges are not equal.")

        # deal with constant bases

        type1 = self.type
        type2 = other.type

        if type1 == 'const' and type2 == 'const':
            prodbasisobj = create_constant_basis(range1)
            return prodbasisobj

        if type1 == 'const':
            return other

        if type2 == 'const':
            return self

        # deal with bspline basis
        # get the number of basis functions
        nbasis1 = self.nbasis
        nbasis2 = other.nbasis
        if type1 == 'bspline' and type2 == 'bspline':

            interiorknots1 = self.params
            interiorknots2 = other.params

            interiorknots12 = np.union1d(interiorknots1, interiorknots2)
            interiorknots12.sort()
            nunique = len(interiorknots12)
            multunique = np.zeros(nunique)

            for i in range(nunique):
                mult1 = interiorknots1 == (interiorknots12[i]) if len(interiorknots1) > 0 else 0
                mult2 = interiorknots2 == (interiorknots12[i]) if len(interiorknots2) > 0 else 0
                multunique[i] = max(sum(mult1), sum(mult2))

            allknots = np.zeros(int(np.sum(multunique)))

            m2 = 0
            for i in range(nunique):
                m1 = m2 + 1
                m2 = int(m2 + multunique[i])
                allknots[m1 - 1:m2] = interiorknots12[i]

            norder1 = nbasis1 - len(interiorknots1)
            norder2 = nbasis2 - len(interiorknots2)
            # norder is not allowed to exceed 20
            norder = min([norder1 + norder2 - 1, 20])
            allbreaks = np.concatenate((np.concatenate((range1[0], allknots), axis=None), range1[1]), axis=None)
            nbasis = len(allbreaks) + norder - 2
            prodbasisobj = create_bspline_basis(rangeval=range1, nbasis=nbasis, norder=norder, breaks=allbreaks)

            return prodbasisobj

        if type1 == 'bspline' or type2 == 'bspline':
            norder = 8
            if type1 == 'bspline':
                interiorknots1 = self.params
                norder1 = nbasis1 - len(interiorknots1)
                norder = min([norder1 + 2, norder])
            if type2 == 'bspline':
                interiorknots2 = other.params
                norder2 = nbasis2 - len(interiorknots2)
                norder = min([norder2 + 2, norder])

            nbasis = max([nbasis1 + nbasis2, norder + 1])
            prodbasisobj = create_bspline_basis(rangeval=range1, nbasis=nbasis, norder=norder)

            return prodbasisobj

    __rmul__ = __mul__



def rangechk(rangeval):
    nrangeval = len(rangeval)
    OK = True

    if rangeval[0] >= rangeval[1]:
        OK = False
    if nrangeval < 1 or nrangeval > 2:
        OK = False
    return OK



