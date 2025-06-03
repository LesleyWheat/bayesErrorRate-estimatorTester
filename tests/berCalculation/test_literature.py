# Desc

# Global imports
import pytest
import numpy as np

# ------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    idlist = []
    argvalues = []
    for scenario in funcarglist:
        idlist.append(scenario[0])
        items = scenario[1].items()
        argnames = [x[0] for x in items]
        argvalues.append([x[1] for x in items])

    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")

# ------------------------------------------------------------------------


class Test_fukunaga():
    '''
    The following values are taken from:
        K. Fukunaga. Introduction to Statastical Pattern Recognition.
        (2nd Ed.), Academic Press, 1990. Pages 45-46.
    Tolerances were not given in the original publication.
    '''
    nFeatures = 8
    seed = 2

    one = {'mu1': np.array([0, 0, 0, 0, 0, 0, 0, 0]),
           'mu2': np.array([2.56, 0, 0, 0, 0, 0, 0, 0]),
           'covar1': np.identity(8),
           'covar2': np.identity(8),
           'ber': 10,
           'tol': 0.5}

    one_flip = {'mu2': np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                'mu1': np.array([2.56, 0, 0, 0, 0, 0, 0, 0]),
                'covar2': np.identity(8),
                'covar1': np.identity(8),
                'ber': 10,
                'tol': 0.5}

    two = {'mu1': np.array([0, 0, 0, 0, 0, 0, 0, 0]),
           'mu2': np.array([0, 0, 0, 0, 0, 0, 0, 0]),
           'covar1': np.identity(8),
           'covar2': 4*np.identity(8),
           'ber': 9,
           'tol': 0.5}

    two_flip = {'mu2': np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                'mu1': np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                'covar2': np.identity(8),
                'covar1': 4*np.identity(8),
                'ber': 9,
                'tol': 0.5}

    c2 = np.array([8.41, 12.06, 0.12, 0.22, 1.49, 1.77, 0.35, 2.73])

    # Note: Case 3 is likely to fail due to numerical precision problems
    # There may be a slight rounding error in the book
    # So inside of 0.05, tolerance is 0.1
    three = {'mu1': np.array([0, 0, 0, 0, 0, 0, 0, 0]),
             'mu2': np.array([3.86, 3.10, 0.84, 0.84, 1.64, 1.08, 0.26, 0.01]),
             'covar1': np.identity(8),
             'covar2': c2 * np.identity(8),
             'ber': 1.9,
             'tol': 0.2}

    three_flip = {'mu2': np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                  'mu1': np.array([3.86, 3.10, 0.84, 0.84, 1.64, 1.08, 0.26, 0.01]),
                  'covar2': np.identity(8),
                  'covar1': c2 * np.identity(8),
                  'ber': 1.9,
                  'tol': 0.2}

    case1 = ("one", {'dct': one})
    case1F = ("oneF", {'dct': one_flip})
    case2 = ("two", {'dct': two})
    case2F = ("two", {'dct': two_flip})
    case3 = ("three", {'dct': three})
    case3F = ("threeF", {'dct': three_flip})

    scenarios = [case1, case2, case3]

    params = {
        "test_bec": [case1, case1F],
        "test_pdf_int": [case1, case1F, case2, case2F, case3, case3F],
        "test_mc_int": [case1, case1F, case2, case2F, case3, case3F],
        "test_mc_intBatchSize": [case1, case1F, case2, case2F, case3, case3F],
        "test_mc_intBatchSize_xfail": [case1, case1F, case2, case2F, case3, case3F],
        'test_calc': [case1, case1F, case2, case2F],
        'test_calc_fail': [case3F]
    }

    @pytest.mark.basic
    @pytest.mark.ber
    def test_bec(self, dct):
        ber = simpleBayesErrorCalc(dct['mu1'], dct['mu2'], dct['covar1'])
        assert ber == pytest.approx(dct['ber'], abs=dct['tol'])

    def _runBatchCheck(self, dct, seed, nLoops, nBatch):
        from distributions._blobs import oneBlob
        from bayesErrorRate import mcBerBatch

        rng = np.random.default_rng(seed=seed)

        distA = oneBlob(self.nFeatures, 0, 1)
        distB = oneBlob(self.nFeatures, 0, 4)
        distA.set_mean(dct['mu1'])
        distB.set_mean(dct['mu2'])
        distA.set_covar(dct['covar1'])
        distB.set_covar(dct['covar2'])

        ber, v = mcBerBatch(rng, [distA, distB],
                            nLoops=nLoops, nBatch=nBatch, brute=False)

        rng = np.random.default_rng(seed=seed)
        ber_mc, v = mcBerBatch(rng, [distA, distB],
                               nLoops=nLoops, nBatch=nBatch, brute=True)

        # Naive bayes
        from sklearn.naive_bayes import GaussianNB
        rng = np.random.default_rng(seed=seed)
        setA = distA.generate(rng, round(nLoops*nBatch/2))
        setB = distB.generate(rng, round(nLoops*nBatch/2))
        X = np.concatenate((setA, setB), axis=0)
        Y = np.repeat([0, 1], [setA.shape[0], setB.shape[0]])
        gnb = GaussianNB()
        y_pred = gnb.fit(X, Y).predict(X)
        nb_acc = (Y != y_pred).sum() / X.shape[0]

        print(f'Evaluated: {ber} Brute force: {ber_mc} Naive bayes: {nb_acc}')

        assert 100*ber == pytest.approx(dct['ber'], abs=dct['tol'])

    @pytest.mark.parametrize("seed", [2, 3, 4, 5])
    @pytest.mark.filterwarnings("ignore::UserWarning")
    # @pytest.mark.parametrize("nint, nBatch", [(1000, 1024), (100, 1024), (50, 2048)])
    @pytest.mark.parametrize("nint, nBatch", [(50, 2048)])
    def test_mc_intBatchSize(self, dct, seed, nint, nBatch):
        self._runBatchCheck(dct, seed, nint, nBatch)

    @pytest.mark.xfail
    @pytest.mark.parametrize("seed", [2, 3, 4, 5])
    @pytest.mark.parametrize("nint, nBatch", [(10, 10*1024), (50, 3*1024)])
    def test_mc_intBatchSize_xfail(self, dct, seed, nint, nBatch):
        self._runBatchCheck(dct, seed, nint, nBatch)

    @pytest.mark.filterwarnings("ignore:divide by zero encountered in double_scalars")
    def test_calc(self, dct):
        ber = complexBayesErrorCalc(
            dct['mu1'], dct['mu2'], dct['covar1'], dct['covar2'])

        ber = ber*100

        assert ber == pytest.approx(dct['ber'], abs=dct['tol'])

    @pytest.mark.xfail
    @pytest.mark.filterwarnings("ignore:divide by zero encountered in double_scalars")
    def test_calc_fail(self, dct):
        ber = complexBayesErrorCalc(
            dct['mu1'], dct['mu2'], dct['covar1'], dct['covar2'])

        ber = ber*100

        assert ber == pytest.approx(dct['ber'], abs=dct['tol'])


# ------------------------------------------------------------------------

def simpleBayesErrorCalc(m1, m2, covar):
    # Pattern Recognition and Neural Networks p21
    from scipy.spatial.distance import mahalanobis
    from scipy.stats import norm

    m1 = m1.reshape((-1))
    m2 = m2.reshape((-1))

    s = mahalanobis(m1, m2, covar)
    ber = 100*norm.cdf(-1*s/2)

    return ber


def complexBayesErrorCalc(m1, m2, cv1, cv2):
    # Introduction to Statistical Pattern Recognition second edition p93

    # Imports
    import warnings
    # warnings.filterwarnings("error")

    # mpmath is better than scipy quad
    import mpmath as mp
    from scipy.linalg import sqrtm

    # number of dimensions
    d = cv1.shape[0]

    # Check inputs
    # cv1 (the first covariance matrix) has to be the identity matrix
    # OR either m1 == m2 or cv1 == cv2 must be true
    # cv1 =/= I AND m1 =/= m2 AND cv1 =/= cv2 will fail
    # I have no idea why
    cond1 = (m1 == m2).all()
    cond2 = (cv1 == cv2).all()
    cond3 = (cv1 == np.identity(d)).all()
    cond4 = (cv2 == np.identity(d)).all() and (cv1 == np.identity(d)).all()
    cond5 = (cv2 == np.identity(d)).all() or (cv1 == np.identity(d)).all()

    flag = True
    if cond3:
        # If first distribution has standard variance, we're good
        flag = False
    elif cond1 & cond2:
        # Can only compare the same distribution in standard case
        flag = True
    elif cond4 & (cond1 ^ cond2):
        # Second distribution is standard and one of mean/covar is different
        flag = False
    elif cond1 & cond5:
        # Means are the same and one of the variances is one
        flag = False

    # flag = False
    if flag:
        raise Exception(
            f" Can't handle inputs: {m1} {m2} {np.max(cv1)} {np.max(cv2)}")

    # MT = M1
    A = sqrtm(cv1)
    U = A.T @ cv2 @ A
    L = A.T @ (m2-m1)

    e1 = complexBayesErrorCalc_stepOne(U, L, d)
    e2 = complexBayesErrorCalc_stepTwo(U, L, d)

    e = e1*0.5+e2*0.5

    print(e1)
    print(e2)
    print(e)

    return float(mp.re(e))


def complexBayesErrorCalc_stepOne(U, L, nFeatures,
                                  tol=-1e-12, maxdegree=10):
    import mpmath as mp
    from numpy.linalg import inv, det

    # from book
    t = 0

    # Duplicate function for local variables
    def F(w):
        f = 1
        for idx in range(nFeatures):
            li = Li[idx, idx]
            vi = V[idx, idx]

            te = 1-1j*w/li

            g = 1/mp.sqrt(te)
            k = mp.exp((-1/2) * mp.power(w, 2) * mp.power(vi, 2) / te)
            f = f*g*k
        f = f*mp.exp(1j*w*c)
        return f

    # Duplicate function for local variables
    def intF(w):
        # imaginary will cancel so we only need real
        r = mp.re(F(w)/(1j * w))
        return r

    Li = np.zeros((nFeatures, nFeatures))
    V = np.zeros((nFeatures, nFeatures))
    for idx in range(nFeatures):
        V[idx, idx] = -L[idx]/U[idx, idx]

        Li[idx, idx] = 1/(1-1/U[idx, idx])

        if U[idx, idx] == 0:
            print('U[idx, idx] == 0')
            print(U[idx, idx])
            print(Li[idx, idx])

    c = (-1/2)*L.T @ inv(U) @ L - (1/2)*mp.log(det(U)) - t
    print(f'1 U: {U} L: {L} V {V} Li {Li} c {c}')

    # F is symetrical, only have to integrate half
    # There is a discontenuity at zero
    int1 = 2*mp.quad(intF, [-1*mp.inf, tol], maxdegree=maxdegree)
    e1 = 1/2 + int1/(2*mp.pi)

    return e1


def complexBayesErrorCalc_stepTwo(U, L, nFeatures,
                                  tol=-1e-12, maxdegree=10):
    import mpmath as mp
    from numpy.linalg import det

    # from book
    t = 0

    # Duplicate function for local variables
    def F(w):
        f = 1
        for idx in range(nFeatures):
            li = Li[idx, idx]
            vi = V[idx, idx]

            te = 1-1j*w/li

            g = 1/mp.sqrt(te)
            k = mp.exp((-1/2) * mp.power(w, 2) * mp.power(vi, 2) / te)
            f = f*g*k
        f = f*mp.exp(1j*w*c)
        return f

    # Duplicate function for local variables
    def intF(w):
        # imaginary will cancel so we only need real
        r = mp.re(F(w)/(1j * w))
        return r

    # MT=M2
    Li = np.zeros((nFeatures, nFeatures))
    V = np.zeros((nFeatures, nFeatures))
    for idx in range(nFeatures):
        V[idx, idx] = -L[idx]*mp.sqrt(U[idx, idx])

        Li[idx, idx] = 1/(U[idx, idx]-1)

        if U[idx, idx] == 1:
            print('U[idx, idx] == 1')
            print(f'{U.shape} {Li.shape}')
            print(U[idx, idx])
            print(Li[idx, idx])

    c = (1/2) * L.T @ L - (1/2)*mp.log(det(U)) - t
    print(f'2 U: {U} L: {L} V {V} Li {Li} c {c}')

    int2 = 2*mp.quad(intF, [-1*mp.inf, tol], maxdegree=maxdegree)
    e2 = 1/2 - int2/(2*mp.pi)

    return e2
