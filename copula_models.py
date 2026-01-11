import numpy as np
from numpy import random
import scipy.stats as stats
from scipy.special import gammaln
import particles
from particles import state_space_models as ssm
from particles import distributions as dists

# =============================================================================
# 1. COMPOSANTS COMMUNS (Transition des facteurs)
# =============================================================================

class NormalTransition(dists.ProbDist):
    """
    Transition AR(1) Gaussienne pour les factor loadings.
    Identique pour tous les modèles.
    """
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

def get_correlation_matrix(C, n):
    """Utilitaire pour calculer la matrice de corrélation R à partir des loadings C."""
    # Sigma = C @ C' + I
    CCt = np.matmul(C, C.transpose(0, 2, 1))
    
    # Diagonale pour normalisation (Sigma_ii = 1 + ||C_i||^2)
    diag_Sigma = 1.0 + np.sum(C**2, axis=-1)
    sqrt_diag = np.sqrt(diag_Sigma)
    
    # Normalisation : R_ij = Sigma_ij / (sqrt(Sigma_ii) * sqrt(Sigma_jj))
    inv_sqrt_diag = 1.0 / sqrt_diag
    normalizer = inv_sqrt_diag[:, :, np.newaxis] * inv_sqrt_diag[:, np.newaxis, :]
    
    R = (CCt + np.eye(n)) * normalizer
    # Force la diagonale à 1.0 pour éviter les erreurs numériques
    I_n = np.eye(n)[np.newaxis, :, :]
    R = R * (1 - I_n) + I_n
    return R

# =============================================================================
# 2. MODÈLE GAUSSIEN (Gaussian Factor Copula)
# =============================================================================

class GaussianCopulaDist(dists.ProbDist):
    def __init__(self, loadings):
        self.C = loadings if loadings.ndim == 3 else loadings[np.newaxis, :, :]
        self.N, self.n, self.p = self.C.shape

    def logpdf(self, x):
        x = np.atleast_1d(x)
        R = get_correlation_matrix(self.C, self.n)
        
        # Log-Vraisemblance Multivariée Normale (Standardisée)
        # log f(x) = -0.5 * log|R| - 0.5 * x' R^-1 x
        sign, log_det = np.linalg.slogdet(R)
        
        # Inversion (ou Cholesky)
        try:
            R_inv = np.linalg.inv(R)
            quad = np.einsum('j,ijk,k->i', x, R_inv, x)
        except np.linalg.LinAlgError:
            return np.full(self.N, -1e10)

        log_const = -0.5 * self.n * np.log(2 * np.pi)
        return log_const - 0.5 * log_det - 0.5 * quad

class GaussianFactorCopulaSSM(ssm.StateSpaceModel):
    def __init__(self, n_series=5, n_factors=1, mu=0.0, phi=0.95, sigma=0.1):
        self.n = n_series
        self.p = n_factors
        self.mu, self.phi, self.sigma = mu, phi, sigma
        self.state_dim = self.n * self.p

    def PX0(self):
        scale = self.sigma / np.sqrt(1 - self.phi**2)
        return NormalTransition(np.full(self.state_dim, self.mu), scale)

    def PX(self, t, xp):
        return NormalTransition(self.mu + self.phi * (xp - self.mu), self.sigma)

    def PY(self, t, xp, x):
        loadings = x.reshape(-1, self.n, self.p) if x.ndim > 1 else x.reshape(1, self.n, self.p)
        return GaussianCopulaDist(loadings)

# =============================================================================
# 3. MODÈLE STUDENT STANDARD (Student-t Factor Copula)
# =============================================================================

class StudentCopulaDist(dists.ProbDist):
    def __init__(self, loadings, nu=5.0):
        self.C = loadings if loadings.ndim == 3 else loadings[np.newaxis, :, :]
        self.nu = nu
        self.N, self.n, self.p = self.C.shape

    def logpdf(self, x):
        x = np.atleast_1d(x)
        R = get_correlation_matrix(self.C, self.n)
        
        sign, log_det = np.linalg.slogdet(R)
        try:
            quad = np.einsum('j,ijk,k->i', x, np.linalg.inv(R), x)
        except np.linalg.LinAlgError:
            return np.full(self.N, -1e10)
        
        # Densité Student Multivariée
        nu, n = self.nu, self.n
        log_const = (gammaln((nu + n) / 2) - gammaln(nu / 2) 
                     - (n / 2.0) * np.log(nu * np.pi))
        log_joint = -((nu + n) / 2) * np.log(1 + quad / nu)
        
        return log_const - 0.5 * log_det + log_joint

class StudentFactorCopulaSSM(ssm.StateSpaceModel):
    def __init__(self, n_series=5, n_factors=1, mu=0.0, phi=0.95, sigma=0.1, nu=5.0):
        self.n = n_series
        self.p = n_factors
        self.mu, self.phi, self.sigma = mu, phi, sigma
        self.nu = nu
        self.state_dim = self.n * self.p

    def PX0(self):
        scale = self.sigma / np.sqrt(1 - self.phi**2)
        return NormalTransition(np.full(self.state_dim, self.mu), scale)

    def PX(self, t, xp):
        return NormalTransition(self.mu + self.phi * (xp - self.mu), self.sigma)

    def PY(self, t, xp, x):
        loadings = x.reshape(-1, self.n, self.p) if x.ndim > 1 else x.reshape(1, self.n, self.p)
        return StudentCopulaDist(loadings, nu=self.nu)

# =============================================================================
# 4. MODÈLE GROUPED STUDENT (Grouped Student-t Factor Copula)
# =============================================================================

class GroupedStudentCopulaDist(dists.ProbDist):
    def __init__(self, loadings, nus, group_map):
        self.C = loadings if loadings.ndim == 3 else loadings[np.newaxis, :, :]
        self.nus = np.array(nus)
        self.group_map = group_map
        self.N, self.n, self.p = self.C.shape

    def logpdf(self, x):
        # NOTE : La densité exacte Grouped Student est complexe à calculer.
        # Pour le filtre particulaire, on utilise une approximation robuste :
        # une Student multivariée avec un nu moyen.
        # Cette approximation capture la structure de corrélation (facteurs)
        # mais simplifie la dépendance de queue pour le calcul des poids.
        
        nu_proxy = np.mean(self.nus) 
        
        # On réutilise la logique Student standard avec nu_proxy
        x = np.atleast_1d(x)
        R = get_correlation_matrix(self.C, self.n)
        
        sign, log_det = np.linalg.slogdet(R)
        try:
            quad = np.einsum('j,ijk,k->i', x, np.linalg.inv(R), x)
        except np.linalg.LinAlgError:
            return np.full(self.N, -1e10)
            
        n = self.n
        log_const = (gammaln((nu_proxy + n) / 2) - gammaln(nu_proxy / 2) 
                     - (n / 2.0) * np.log(nu_proxy * np.pi))
        log_joint = -((nu_proxy + n) / 2) * np.log(1 + quad / nu_proxy)
        
        return log_const - 0.5 * log_det + log_joint

class GroupedStudentFactorCopulaSSM(ssm.StateSpaceModel):
    def __init__(self, n_series=5, n_factors=1, n_groups=2, mu=0.0, phi=0.95, sigma=0.1, nus=None):
        self.n = n_series
        self.p = n_factors
        self.mu, self.phi, self.sigma = mu, phi, sigma
        self.G = n_groups
        self.group_map = np.array([i % self.G for i in range(self.n)])
        self.nus = np.array(nus) if nus is not None else np.array([5.0 + 2*g for g in range(self.G)])
        self.state_dim = self.n * self.p

    def PX0(self):
        scale = self.sigma / np.sqrt(1 - self.phi**2)
        return NormalTransition(np.full(self.state_dim, self.mu), scale)

    def PX(self, t, xp):
        return NormalTransition(self.mu + self.phi * (xp - self.mu), self.sigma)

    def PY(self, t, xp, x):
        loadings = x.reshape(-1, self.n, self.p) if x.ndim > 1 else x.reshape(1, self.n, self.p)
        return GroupedStudentCopulaDist(loadings, self.nus, self.group_map)

# =============================================================================
# 5. FONCTION D'APPEL SIMPLIFIÉE
# =============================================================================

def simulate_copula_model_particles(model, T):
    """
    Simule le modèle en utilisant le moteur interne de 'particles'.
    """
    true_states, observations = model.simulate(T)
    return np.array(true_states), np.array(observations)