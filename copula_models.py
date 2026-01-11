import numpy as np
from numpy import random
import scipy.stats as stats
from scipy.special import gammaln
import math
from numba import njit

import particles
from particles import state_space_models as ssm
from particles import distributions as dists

# =============================================================================
# 1. MOTEUR DE CALCUL OPTIMISÉ (NUMBA)
# =============================================================================

@njit(fastmath=True, cache=True)
def _fast_student_logpdf(x, C, nu):
    """
    Calcul optimisé de la log-densité Student Multivariée pour N particules.
    """
    N, n, p = C.shape
    x_val = x.ravel()
    log_pdfs = np.zeros(N)
    
    # Constantes Gamma
    val_gamma1 = math.lgamma((nu + n) / 2.0)
    val_gamma2 = math.lgamma(nu / 2.0)
    log_const = val_gamma1 - val_gamma2 - (n / 2.0) * np.log(nu * np.pi)
    
    for i in range(N):
        # 1. Construction R (Corrélation)
        diag_inv_sqrt = np.empty(n)
        for k in range(n):
            sum_sq = 0.0
            for l in range(p):
                sum_sq += C[i, k, l]**2
            diag_inv_sqrt[k] = 1.0 / np.sqrt(1.0 + sum_sq)
            
        R = np.eye(n)
        for j in range(n):
            for k in range(j + 1, n):
                sigma_jk = 0.0
                for l in range(p):
                    sigma_jk += C[i, j, l] * C[i, k, l]
                val = sigma_jk * diag_inv_sqrt[j] * diag_inv_sqrt[k]
                R[j, k] = val
                R[k, j] = val

        # 2. Déterminant et Inverse
        det_R = np.linalg.det(R)
        if det_R <= 0:
            log_pdfs[i] = -1e10
            continue
            
        log_det = np.log(det_R)
        R_inv = np.linalg.inv(R)
        
        # 3. Forme quadratique
        tmp = R_inv @ x_val
        quad = np.dot(x_val, tmp)
        
        # 4. Densité
        log_joint = -((nu + n) / 2.0) * np.log(1.0 + quad / nu)
        log_pdfs[i] = log_const - 0.5 * log_det + log_joint
        
    return log_pdfs

# =============================================================================
# 2. TRANSITION D'ÉTAT
# =============================================================================

class NormalTransition(dists.ProbDist):
    def __init__(self, loc, scale):
        self.loc = np.atleast_1d(loc)
        self.scale = scale
    
    @property
    def dim(self):
        return self.loc.shape[-1]
    
    def logpdf(self, x):
        return np.sum(stats.norm.logpdf(x, loc=self.loc, scale=self.scale), axis=-1)
    
    def rvs(self, size=None):
        if self.loc.ndim == 1:
            d = len(self.loc)
            if size is None: return random.normal(loc=self.loc, scale=self.scale)
            return random.normal(loc=self.loc, scale=self.scale, size=(size, d))
        return random.normal(loc=self.loc, scale=self.scale)

# =============================================================================
# 3. MODÈLE GAUSSIEN
# =============================================================================

class GaussianCopulaDist(dists.ProbDist):
    def __init__(self, loadings):
        self.C = loadings
        self.N, self.n, self.p = self.C.shape

    def logpdf(self, x):
        # Fallback NumPy pour le Gaussien (moins critique ou à optimiser si besoin)
        x = np.atleast_1d(x)
        CCt = np.matmul(self.C, self.C.transpose(0, 2, 1))
        diag_Sigma = 1.0 + np.sum(self.C**2, axis=-1)
        inv_sqrt_diag = 1.0 / np.sqrt(diag_Sigma)
        normalizer = inv_sqrt_diag[:, :, np.newaxis] * inv_sqrt_diag[:, np.newaxis, :]
        R = (CCt + np.eye(self.n)) * normalizer
        I_n = np.eye(self.n)[np.newaxis, :, :]
        R = R * (1 - I_n) + I_n
        
        sign, log_det = np.linalg.slogdet(R)
        try:
            R_inv = np.linalg.inv(R)
            quad = np.einsum('j,ijk,k->i', x, R_inv, x)
        except np.linalg.LinAlgError:
            return np.full(self.N, -1e10)
        return -0.5 * self.n * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * quad

    def rvs(self, size=None):
        C = self.C[0]
        z_factors = random.normal(0, 1, size=self.p)
        z_idio = random.normal(0, 1, size=self.n)
        Z_gaussian = C @ z_factors + z_idio
        var_marginal = 1 + np.sum(C**2, axis=1)
        return Z_gaussian / np.sqrt(var_marginal)

class GaussianFactorCopulaSSM(ssm.StateSpaceModel):
    def __init__(self, n_series=5, n_factors=1, mu=0.0, phi=0.95, sigma=0.1):
        self.n, self.p = n_series, n_factors
        self.mu, self.phi, self.sigma = mu, phi, sigma
        self.state_dim = self.n * self.p

    def PX0(self):
        scale = self.sigma / np.sqrt(1 - self.phi**2)
        return NormalTransition(np.full(self.state_dim, self.mu), scale)

    def PX(self, t, xp):
        return NormalTransition(self.mu + self.phi * (xp - self.mu), self.sigma)

    def PY(self, t, xp, x):
        loadings = x.reshape(-1, self.n, self.p)
        return GaussianCopulaDist(loadings)

# =============================================================================
# 4. MODÈLE STUDENT STANDARD (OPTIMISÉ)
# =============================================================================

class StudentCopulaDist(dists.ProbDist):
    def __init__(self, loadings, nu=5.0):
        self.C = loadings
        self.nu = nu
        self.N, self.n, self.p = self.C.shape

    def logpdf(self, x):
        x_arr = np.atleast_1d(x).astype(np.float64)
        return _fast_student_logpdf(x_arr, self.C, self.nu)

    def rvs(self, size=None):
        C = self.C[0]
        z_factors = random.normal(0, 1, size=self.p)
        z_idio = random.normal(0, 1, size=self.n)
        Z_gaussian = C @ z_factors + z_idio
        w = random.gamma(shape=self.nu/2.0, scale=2.0/self.nu)
        x_raw = np.sqrt(1.0/w) * Z_gaussian
        var_marginal = 1 + np.sum(C**2, axis=1)
        return x_raw / np.sqrt(var_marginal)

class StudentFactorCopulaSSM(ssm.StateSpaceModel):
    def __init__(self, n_series=5, n_factors=1, mu=0.0, phi=0.95, sigma=0.1, nu=5.0):
        self.n, self.p = n_series, n_factors
        self.mu, self.phi, self.sigma = mu, phi, sigma
        self.nu = nu
        self.state_dim = self.n * self.p

    def PX0(self):
        scale = self.sigma / np.sqrt(1 - self.phi**2)
        return NormalTransition(np.full(self.state_dim, self.mu), scale)

    def PX(self, t, xp):
        return NormalTransition(self.mu + self.phi * (xp - self.mu), self.sigma)

    def PY(self, t, xp, x):
        loadings = x.reshape(-1, self.n, self.p)
        return StudentCopulaDist(loadings, nu=self.nu)

# =============================================================================
# 5. MODÈLE GROUPED STUDENT (OPTIMISÉ)
# =============================================================================

class GroupedStudentCopulaDist(dists.ProbDist):
    def __init__(self, loadings, nus, group_map):
        self.C = loadings
        self.nus, self.group_map = np.array(nus), group_map
        self.N, self.n, self.p = self.C.shape

    def logpdf(self, x):
        x_arr = np.atleast_1d(x).astype(np.float64)
        return _fast_student_logpdf(x_arr, self.C, np.mean(self.nus))

    def rvs(self, size=None):
        C = self.C[0]
        z_factors = random.normal(0, 1, size=self.p)
        z_idio = random.normal(0, 1, size=self.n)
        Z_gaussian = C @ z_factors + z_idio
        x_raw = np.zeros(self.n)
        zetas = {}
        for g in np.unique(self.group_map):
            nu_g = self.nus[g]
            zetas[g] = 1.0 / random.gamma(shape=nu_g/2.0, scale=2.0/nu_g)
        for i in range(self.n):
            x_raw[i] = np.sqrt(zetas[self.group_map[i]]) * Z_gaussian[i]
        var_marginal = 1 + np.sum(C**2, axis=1)
        return x_raw / np.sqrt(var_marginal)

class GroupedStudentFactorCopulaSSM(ssm.StateSpaceModel):
    def __init__(self, n_series=5, n_factors=1, n_groups=2, mu=0.0, phi=0.95, sigma=0.1, nus=None):
        self.n, self.p = n_series, n_factors
        self.mu, self.phi, self.sigma = mu, phi, sigma
        self.G, self.group_map = n_groups, np.array([i % n_groups for i in range(n_series)])
        self.nus = np.array(nus) if nus is not None else np.array([5.0 + 2*g for g in range(n_groups)])
        self.state_dim = self.n * self.p

    def PX0(self):
        scale = self.sigma / np.sqrt(1 - self.phi**2)
        return NormalTransition(np.full(self.state_dim, self.mu), scale)

    def PX(self, t, xp):
        return NormalTransition(self.mu + self.phi * (xp - self.mu), self.sigma)

    def PY(self, t, xp, x):
        loadings = x.reshape(-1, self.n, self.p)
        return GroupedStudentCopulaDist(loadings, self.nus, self.group_map)

# =============================================================================
# 6. WRAPPER DE SIMULATION
# =============================================================================

def simulate_copula_model_particles(model, T):
    true_states, observations = model.simulate(T)
    return np.array(true_states), np.array(observations)