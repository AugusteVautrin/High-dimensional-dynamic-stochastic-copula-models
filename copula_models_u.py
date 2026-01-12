"""
Copules Factorielles Dynamiques - Version avec Uniformes u_t
============================================================
Implémentation fidèle à Creal & Tsay (2015)

La densité de copule est (équation 8):
    c(u_t | Λ_t) = p(x_t | Λ_t) / ∏_i p(x_it)

où x_it = P^{-1}(u_it) est la transformation inverse des uniformes.

Pour la copule Gaussienne:
    c(u) = |R|^{-1/2} exp(-1/2 x'(R^{-1} - I)x)

Pour la copule Student:
    c(u) = f_{ν,n}(x; R) / ∏_i f_ν(x_i)
"""

import numpy as np
from numpy import random
import scipy.stats as stats
from scipy.special import gammaln
from scipy.stats import norm, t as student_t
import math
from numba import njit

import particles
from particles import state_space_models as ssm
from particles import distributions as dists



@njit(fastmath=True, cache=True)
def _build_correlation_matrix(C, n, p):
    """
    Construit la matrice de corrélation R à partir des loadings C.
    R_ij = (C_i · C_j) / sqrt((1 + ||C_i||²)(1 + ||C_j||²))
    """
    # Calcul de 1 + ||C_i||²
    diag_sigma = np.empty(n)
    for i in range(n):
        sum_sq = 0.0
        for l in range(p):
            sum_sq += C[i, l] ** 2
        diag_sigma[i] = 1.0 + sum_sq
    
    # Construction de R
    R = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            # C_i · C_j
            dot_ij = 0.0
            for l in range(p):
                dot_ij += C[i, l] * C[j, l]
            
            # Normalisation
            R[i, j] = dot_ij / np.sqrt(diag_sigma[i] * diag_sigma[j])
            R[j, i] = R[i, j]
    
    return R


@njit(fastmath=True, cache=True)
def _gaussian_copula_logpdf(x, C, N, n, p):
    """
    Log-densité de la COPULE Gaussienne (pas la densité MVN).
    
    c(u) = |R|^{-1/2} exp(-1/2 x'(R^{-1} - I)x)
    
    où x = Φ^{-1}(u)
    
    log c(u) = -1/2 log|R| - 1/2 x'R^{-1}x + 1/2 x'x
    """
    log_pdfs = np.zeros(N)
    
    for i in range(N):
        # Construction de R pour cette particule
        R = _build_correlation_matrix(C[i], n, p)
        
        # Déterminant
        det_R = np.linalg.det(R)
        if det_R <= 1e-10:
            log_pdfs[i] = -1e10
            continue
        
        log_det = np.log(det_R)
        
        # Inverse
        R_inv = np.linalg.inv(R)
        
        # Formes quadratiques
        # x' R^{-1} x
        quad_R_inv = 0.0
        for j in range(n):
            for k in range(n):
                quad_R_inv += x[j] * R_inv[j, k] * x[k]
        
        # x' x
        quad_I = 0.0
        for j in range(n):
            quad_I += x[j] ** 2
        
        # Densité de copule
        log_pdfs[i] = -0.5 * log_det - 0.5 * quad_R_inv + 0.5 * quad_I
    
    return log_pdfs


@njit(fastmath=True, cache=True)
def _student_copula_logpdf(x, C, nu, N, n, p):
    """
    Log-densité de la COPULE Student (pas la densité MVT).
    
    c(u) = f_{ν,n}(x; R) / ∏_i f_ν(x_i)
    
    où x = T_ν^{-1}(u)
    
    La densité jointe Student multivariée est:
    f_{ν,n}(x; R) = Γ((ν+n)/2) / (Γ(ν/2) (νπ)^{n/2} |R|^{1/2}) * (1 + x'R^{-1}x/ν)^{-(ν+n)/2}
    
    La densité marginale Student univariée est:
    f_ν(x_i) = Γ((ν+1)/2) / (Γ(ν/2) √(νπ)) * (1 + x_i²/ν)^{-(ν+1)/2}
    """
    log_pdfs = np.zeros(N)
    
    # Constantes pour la densité jointe
    log_const_joint = (math.lgamma((nu + n) / 2.0) 
                       - math.lgamma(nu / 2.0) 
                       - (n / 2.0) * np.log(nu * np.pi))
    
    # Constantes pour les marginales (n termes identiques)
    log_const_marginal = (math.lgamma((nu + 1) / 2.0) 
                          - math.lgamma(nu / 2.0) 
                          - 0.5 * np.log(nu * np.pi))
    
    for i in range(N):
        # Construction de R pour cette particule
        R = _build_correlation_matrix(C[i], n, p)
        
        # Déterminant
        det_R = np.linalg.det(R)
        if det_R <= 1e-10:
            log_pdfs[i] = -1e10
            continue
        
        log_det = np.log(det_R)
        
        # Inverse
        R_inv = np.linalg.inv(R)
        
        # Forme quadratique x' R^{-1} x
        quad = 0.0
        for j in range(n):
            for k in range(n):
                quad += x[j] * R_inv[j, k] * x[k]
        
        log_joint = (log_const_joint 
                     - 0.5 * log_det 
                     - ((nu + n) / 2.0) * np.log(1.0 + quad / nu))
        
        log_marginals = 0.0
        for j in range(n):
            log_marginals += (log_const_marginal 
                              - ((nu + 1) / 2.0) * np.log(1.0 + x[j]**2 / nu))
        
        log_pdfs[i] = log_joint - log_marginals
    
    return log_pdfs


@njit(fastmath=True, cache=True)
def _grouped_student_copula_logpdf(x, C, nus, group_map, N, n, p, G):
    """
    Log-densité de la COPULE Grouped Student.
    
    Approximation : on utilise la moyenne des ν pour la densité jointe,
    mais les marginales utilisent le ν de chaque groupe.
    
    Note: La densité exacte est plus complexe car les marginales bivariées
    inter-groupes ne sont pas Student standard.
    """
    log_pdfs = np.zeros(N)
    
    # ν moyen pour la jointe
    nu_mean = 0.0
    for g in range(G):
        nu_mean += nus[g]
    nu_mean /= G
    
    # Constantes pour la densité jointe
    log_const_joint = (math.lgamma((nu_mean + n) / 2.0) 
                       - math.lgamma(nu_mean / 2.0) 
                       - (n / 2.0) * np.log(nu_mean * np.pi))
    
    for i in range(N):
        R = _build_correlation_matrix(C[i], n, p)
        
        det_R = np.linalg.det(R)
        if det_R <= 1e-10:
            log_pdfs[i] = -1e10
            continue
        
        log_det = np.log(det_R)
        R_inv = np.linalg.inv(R)
        
        # Forme quadratique
        quad = 0.0
        for j in range(n):
            for k in range(n):
                quad += x[j] * R_inv[j, k] * x[k]
        
        # Log-densité jointe (avec ν moyen)
        log_joint = (log_const_joint 
                     - 0.5 * log_det 
                     - ((nu_mean + n) / 2.0) * np.log(1.0 + quad / nu_mean))
        
        # Log-densité des marginales (avec ν du groupe)
        log_marginals = 0.0
        for j in range(n):
            g = group_map[j]
            nu_g = nus[g]
            log_const_g = (math.lgamma((nu_g + 1) / 2.0) 
                           - math.lgamma(nu_g / 2.0) 
                           - 0.5 * np.log(nu_g * np.pi))
            log_marginals += log_const_g - ((nu_g + 1) / 2.0) * np.log(1.0 + x[j]**2 / nu_g)
        
        log_pdfs[i] = log_joint - log_marginals
    
    return log_pdfs



class NormalTransition(dists.ProbDist):
    """Transition AR(1) Gaussienne pour les factor loadings."""
    
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
            if size is None:
                return random.normal(loc=self.loc, scale=self.scale)
            return random.normal(loc=self.loc, scale=self.scale, size=(size, d))
        return random.normal(loc=self.loc, scale=self.scale)


class GaussianCopulaDistU(dists.ProbDist):
    """
    Distribution de copule Gaussienne.
    Entrée : u_t ∈ (0,1)^n (uniformes)
    """
    
    def __init__(self, loadings):
        self.C = loadings
        self.N, self.n, self.p = self.C.shape

    def logpdf(self, u):
        """
        Log-densité de copule évaluée en u.
        Transforme u → x = Φ^{-1}(u), puis calcule c(u).
        """
        u = np.atleast_1d(u)
        u = np.clip(u, 1e-8, 1 - 1e-8)
        
        # Transformation inverse
        x = norm.ppf(u)
        
        return _gaussian_copula_logpdf(x, self.C, self.N, self.n, self.p)

    def rvs(self, size=None):
        """Simule u ~ C(·)"""
        C = self.C[0]
        
        # Génération Gaussienne
        z_factors = random.normal(0, 1, size=self.p)
        z_idio = random.normal(0, 1, size=self.n)
        Z_gaussian = C @ z_factors + z_idio
        
        # Standardisation
        var_marginal = 1 + np.sum(C**2, axis=1)
        x = Z_gaussian / np.sqrt(var_marginal)
        
        # Transformation en uniforme
        u = norm.cdf(x)
        
        return u


class StudentCopulaDistU(dists.ProbDist):
    """
    Distribution de copule Student.
    Entrée : u_t ∈ (0,1)^n (uniformes)
    """
    
    def __init__(self, loadings, nu=5.0):
        self.C = loadings
        self.nu = nu
        self.N, self.n, self.p = self.C.shape

    def logpdf(self, u):
        """
        Log-densité de copule évaluée en u.
        Transforme u → x = T_ν^{-1}(u), puis calcule c(u).
        """
        u = np.atleast_1d(u)
        u = np.clip(u, 1e-8, 1 - 1e-8)
        
        # Transformation inverse Student
        x = student_t.ppf(u, df=self.nu).astype(np.float64)
        
        return _student_copula_logpdf(x, self.C, self.nu, self.N, self.n, self.p)

    def rvs(self, size=None):
        """Simule u ~ C(·)"""
        C = self.C[0]
        
        # Génération Student via mélange
        z_factors = random.normal(0, 1, size=self.p)
        z_idio = random.normal(0, 1, size=self.n)
        Z_gaussian = C @ z_factors + z_idio
        
        # Variable de mélange
        w = random.gamma(shape=self.nu/2.0, scale=2.0/self.nu)
        x_raw = np.sqrt(1.0/w) * Z_gaussian
        
        # Standardisation
        var_marginal = 1 + np.sum(C**2, axis=1)
        x = x_raw / np.sqrt(var_marginal)
        
        # Transformation en uniforme via CDF Student
        u = student_t.cdf(x, df=self.nu)
        
        return u


class GroupedStudentCopulaDistU(dists.ProbDist):
    """
    Distribution de copule Grouped Student.
    Entrée : u_t ∈ (0,1)^n (uniformes)
    """
    
    def __init__(self, loadings, nus, group_map):
        self.C = loadings
        self.nus = np.array(nus, dtype=np.float64)
        self.group_map = np.array(group_map, dtype=np.int32)
        self.N, self.n, self.p = self.C.shape
        self.G = len(nus)

    def logpdf(self, u):
        """
        Log-densité de copule évaluée en u.
        Transforme u_i → x_i = T_{ν_g(i)}^{-1}(u_i) avec ν du groupe.
        """
        u = np.atleast_1d(u)
        u = np.clip(u, 1e-8, 1 - 1e-8)
        
        # Transformation inverse avec ν spécifique à chaque groupe
        x = np.zeros(self.n, dtype=np.float64)
        for i in range(self.n):
            g = self.group_map[i]
            x[i] = student_t.ppf(u[i], df=self.nus[g])
        
        return _grouped_student_copula_logpdf(
            x, self.C, self.nus, self.group_map, 
            self.N, self.n, self.p, self.G
        )

    def rvs(self, size=None):
        """Simule u ~ C(·)"""
        C = self.C[0]
        
        # Génération Gaussienne de base
        z_factors = random.normal(0, 1, size=self.p)
        z_idio = random.normal(0, 1, size=self.n)
        Z_gaussian = C @ z_factors + z_idio
        
        # Variables de mélange par groupe
        zetas = {}
        for g in range(self.G):
            nu_g = self.nus[g]
            zetas[g] = 1.0 / random.gamma(shape=nu_g/2.0, scale=2.0/nu_g)
        
        # Application du mélange
        x_raw = np.zeros(self.n)
        for i in range(self.n):
            g = self.group_map[i]
            x_raw[i] = np.sqrt(zetas[g]) * Z_gaussian[i]
        
        # Standardisation
        var_marginal = 1 + np.sum(C**2, axis=1)
        x = x_raw / np.sqrt(var_marginal)
        
        # Transformation en uniforme avec CDF du groupe
        u = np.zeros(self.n)
        for i in range(self.n):
            g = self.group_map[i]
            u[i] = student_t.cdf(x[i], df=self.nus[g])
        
        return u
    

class GaussianFactorCopulaSSM_U(ssm.StateSpaceModel):
    """
    Modèle Gaussien Factor Copula avec loadings dynamiques.
    Observation : u_t ∈ (0,1)^n
    """
    
    def __init__(self, n_series=5, n_factors=1, mu=0.0, phi=0.95, sigma=0.1):
        self.n = n_series
        self.p = n_factors
        self.mu = mu
        self.phi = phi
        self.sigma = sigma
        self.state_dim = self.n * self.p

    def PX0(self):
        scale = self.sigma / np.sqrt(1 - self.phi**2)
        return NormalTransition(np.full(self.state_dim, self.mu), scale)

    def PX(self, t, xp):
        return NormalTransition(self.mu + self.phi * (xp - self.mu), self.sigma)

    def PY(self, t, xp, x):
        loadings = x.reshape(-1, self.n, self.p)
        return GaussianCopulaDistU(loadings)


class StudentFactorCopulaSSM_U(ssm.StateSpaceModel):
    """
    Modèle Student-t Factor Copula avec loadings dynamiques.
    Observation : u_t ∈ (0,1)^n
    """
    
    def __init__(self, n_series=5, n_factors=1, mu=0.0, phi=0.95, sigma=0.1, nu=5.0):
        self.n = n_series
        self.p = n_factors
        self.mu = mu
        self.phi = phi
        self.sigma = sigma
        self.nu = nu
        self.state_dim = self.n * self.p

    def PX0(self):
        scale = self.sigma / np.sqrt(1 - self.phi**2)
        return NormalTransition(np.full(self.state_dim, self.mu), scale)

    def PX(self, t, xp):
        return NormalTransition(self.mu + self.phi * (xp - self.mu), self.sigma)

    def PY(self, t, xp, x):
        loadings = x.reshape(-1, self.n, self.p)
        return StudentCopulaDistU(loadings, nu=self.nu)


class GroupedStudentFactorCopulaSSM_U(ssm.StateSpaceModel):
    """
    Modèle Grouped Student-t Factor Copula avec loadings dynamiques.
    Observation : u_t ∈ (0,1)^n
    """
    
    def __init__(self, n_series=5, n_factors=1, n_groups=2, mu=0.0, phi=0.95, sigma=0.1, nus=None):
        self.n = n_series
        self.p = n_factors
        self.mu = mu
        self.phi = phi
        self.sigma = sigma
        self.G = n_groups
        self.group_map = np.array([i % n_groups for i in range(n_series)])
        self.nus = np.array(nus) if nus is not None else np.array([5.0 + 2*g for g in range(n_groups)])
        self.state_dim = self.n * self.p

    def PX0(self):
        scale = self.sigma / np.sqrt(1 - self.phi**2)
        return NormalTransition(np.full(self.state_dim, self.mu), scale)

    def PX(self, t, xp):
        return NormalTransition(self.mu + self.phi * (xp - self.mu), self.sigma)

    def PY(self, t, xp, x):
        loadings = x.reshape(-1, self.n, self.p)
        return GroupedStudentCopulaDistU(loadings, self.nus, self.group_map)
    

def simulate_copula_model_U(model, T):
    """
    Simule le modèle et retourne les états et les uniformes.
    
    Retourne
    --------
    states : array (T, state_dim)
        États latents (loadings)
    u_data : array (T, n)
        Observations uniformes u_t ∈ (0,1)^n
    """
    states, observations = model.simulate(T)
    return np.array(states), np.array(observations)


def u_to_x_gaussian(u):
    """Convertit uniformes → x pour copule Gaussienne."""
    u = np.clip(u, 1e-8, 1 - 1e-8)
    return norm.ppf(u)


def u_to_x_student(u, nu):
    """Convertit uniformes → x pour copule Student."""
    u = np.clip(u, 1e-8, 1 - 1e-8)
    return student_t.ppf(u, df=nu)


def x_to_u_gaussian(x):
    """Convertit x → uniformes pour copule Gaussienne."""
    return norm.cdf(x)


def x_to_u_student(x, nu):
    """Convertit x → uniformes pour copule Student."""
    return student_t.cdf(x, df=nu)


def compute_correlation_from_loadings(lambdas):
    """
    Calcule la matrice de corrélation conditionnelle R_t à partir des loadings.
    
    R_ij = (λ_i' λ_j) / sqrt((1 + ||λ_i||²)(1 + ||λ_j||²))
    
    Paramètres
    ----------
    lambdas : array (T, n, p) ou (n, p)
        Factor loadings
    
    Retourne
    --------
    R : array (T, n, n) ou (n, n)
        Matrices de corrélation
    """
    if lambdas.ndim == 2:
        lambdas = lambdas[np.newaxis, :, :]
    
    T, n, p = lambdas.shape
    R = np.zeros((T, n, n))
    
    for t in range(T):
        lam = lambdas[t]  # (n, p)
        lam_sq = np.sum(lam**2, axis=1)  # ||λ_i||²
        
        # Numérateur : λ_i' λ_j
        numerator = lam @ lam.T
        
        # Dénominateur : sqrt((1 + ||λ_i||²)(1 + ||λ_j||²))
        denominator = np.sqrt(np.outer(1 + lam_sq, 1 + lam_sq))
        
        R[t] = numerator / denominator
        np.fill_diagonal(R[t], 1.0)
    
    return R.squeeze()


def compute_kendall_tau(R):
    """
    Calcule le tau de Kendall à partir de la corrélation linéaire.
    
    Pour Gaussien et Student : τ = (2/π) arcsin(ρ)
    
    Paramètres
    ----------
    R : array (..., n, n)
        Matrices de corrélation
    
    Retourne
    --------
    tau : array (..., n, n)
        Matrices de tau de Kendall
    """
    return (2 / np.pi) * np.arcsin(R)