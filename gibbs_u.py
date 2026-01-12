import numpy as np
from scipy import stats, linalg, special
from scipy.stats import norm, t as student_t
import particles
from particles import state_space_models as ssm
from particles import distributions as dists
from particles import mcmc
from numba import njit
import math
from tqdm import tqdm

# =============================================================================
# 1. MOTEUR DE CALCUL ULTRA-RAPIDE (NUMBA)
# =============================================================================

@njit(fastmath=True, cache=True)
def _numba_copula_likelihood(x_proposals, i, n, p, z_t, zeta_t, x_data_t, all_lambdas_t, nu, is_student):
    """Calcul optimisé de la log-vraisemblance pour les particules."""
    N = x_proposals.shape[0]
    log_pdfs = np.empty(N)
    
    if is_student:
        log_const_joint = (math.lgamma((nu + n) / 2.0) - math.lgamma(nu / 2.0) 
                          - (n / 2.0) * np.log(nu * np.pi))
        log_const_marg = (math.lgamma((nu + 1) / 2.0) - math.lgamma(nu / 2.0) 
                         - 0.5 * np.log(nu * np.pi))
    
    for m in range(N):
        current_lambdas = all_lambdas_t.copy()
        current_lambdas[i, 0] = x_proposals[m]
        
        # Construction de R_t (Corrélation)
        R = np.eye(n)
        norm_sq = np.zeros(n)
        for k in range(n):
            for l in range(p):
                norm_sq[k] += current_lambdas[k, l] ** 2
        
        for row in range(n):
            for col in range(row + 1, n):
                dot = 0.0
                for l in range(p):
                    dot += current_lambdas[row, l] * current_lambdas[col, l]
                val = dot / np.sqrt((1.0 + norm_sq[row]) * (1.0 + norm_sq[col]))
                R[row, col] = R[col, row] = val

        det_R = np.linalg.det(R)
        if det_R <= 1e-10:
            log_pdfs[m] = -1e10
            continue
            
        R_inv = np.linalg.inv(R)
        quad = 0.0
        for r in range(n):
            for c in range(n):
                quad += x_data_t[r] * R_inv[r, c] * x_data_t[c]
        
        if is_student:
            log_joint = log_const_joint - 0.5 * np.log(det_R) - ((nu + n) / 2.0) * np.log(1.0 + quad / (nu * zeta_t))
            log_marginals = 0.0
            for j in range(n):
                log_marginals += log_const_marg - ((nu + 1) / 2.0) * np.log(1.0 + x_data_t[j]**2 / (nu * zeta_t))
            log_pdfs[m] = log_joint - log_marginals
        else:
            quad_I = 0.0
            for j in range(n): quad_I += x_data_t[j]**2
            log_pdfs[m] = -0.5 * np.log(det_R) - 0.5 * quad + 0.5 * quad_I
            
    return log_pdfs

# =============================================================================
# 2. CLASSES DU MODÈLE POUR PARTICLES
# =============================================================================

class CopulaLikelihood(dists.ProbDist):
    def __init__(self, i, t, n, p, z_t, zeta_t, x_data_t, all_lambdas_t, nu=None):
        self.params = (i, n, p, z_t, zeta_t, x_data_t, all_lambdas_t, nu, nu is not None)
    def logpdf(self, x):
        return _numba_copula_likelihood(np.atleast_1d(x).astype(np.float64), *self.params)

class LoadingSSM(ssm.StateSpaceModel):
    def __init__(self, i, n, p, mu, phi, sigma, z, zeta, x_data, all_lambdas, nu=None):
        self.i, self.n, self.p, self.mu, self.phi, self.sigma = i, n, p, mu, phi, sigma
        self.z, self.zeta, self.x_data, self.all_lambdas, self.nu = z, zeta, x_data, all_lambdas, nu
    def PX0(self): return dists.Normal(loc=self.mu, scale=self.sigma * 10.0)
    def PX(self, t, xp): return dists.Normal(loc=self.mu + self.phi * (xp - self.mu), scale=self.sigma)
    def PY(self, t, xp, x):
        return CopulaLikelihood(self.i, t, self.n, self.p, self.z[t], 
                               self.zeta[t] if self.nu else 1.0, 
                               self.x_data[t], self.all_lambdas[t], self.nu)

# =============================================================================
# 3. GIBBS SAMPLER COMPLET
# =============================================================================

class FactorCopulaGibbs:
    def __init__(self, u_data, n_factors=1, copula_type='student', nu_init=8.0):
        self.u_data = np.clip(u_data, 1e-7, 1-1e-7)
        self.T, self.n = u_data.shape
        self.p = n_factors
        self.is_student = (copula_type == 'student')
        self.nu = nu_init if self.is_student else None
        self.update_x_data()
        
        self.theta = {'mu': np.full((self.n, self.p), 0.4),
                      'phi': np.full((self.n, self.p), 0.97),
                      'sigma': np.full((self.n, self.p), 0.08)}
        
        self.z = np.random.normal(0, 1, (self.T, self.p))
        self.zeta = np.ones(self.T)
        self.lambdas = np.random.normal(0.4, 0.1, (self.T, self.n, self.p))
        self.first_iter = True
        self.trace = {'nu': [], 'mu': [], 'phi': [], 'sigma': []}

    def update_x_data(self):
        self.x_data = student_t.ppf(self.u_data, df=self.nu) if self.is_student else norm.ppf(self.u_data)

    def update_z(self):
        """Étape 1 : Update facteurs latents Z."""
        I_p = np.eye(self.p)
        for t in range(self.T):
            lam_t = self.lambdas[t]
            x_t = self.x_data[t] / (np.sqrt(self.zeta[t]) if self.is_student else 1.0)
            lam_sq = np.sum(lam_t**2, axis=1)
            sigma_sq = 1.0 / (1.0 + lam_sq)
            lam_tilde = lam_t / np.sqrt(1 + lam_sq)[:, np.newaxis]
            D_inv = np.diag(1.0 / sigma_sq)
            prec_post = I_p + lam_tilde.T @ D_inv @ lam_tilde
            L = linalg.cholesky(prec_post, lower=True)
            mu_post = linalg.cho_solve((L, True), lam_tilde.T @ D_inv @ x_t)
            self.z[t] = mu_post + linalg.solve_triangular(L.T, np.random.normal(size=self.p), lower=False)

    def update_lambdas_pg(self, n_particles):
        """Étape 2 : Particle Gibbs (CSMC)."""
        for i in range(self.n):
            ssm_i = LoadingSSM(i, self.n, self.p, self.theta['mu'][i,0], 
                              self.theta['phi'][i,0], self.theta['sigma'][i,0],
                              self.z, self.zeta, self.x_data, self.lambdas, self.nu)
            fk = ssm.Bootstrap(ssm=ssm_i, data=np.zeros(self.T))
            if self.first_iter:
                pf = particles.SMC(fk=fk, N=n_particles, store_history=True, verbose=False)
            else:
                pf = mcmc.CSMC(fk=fk, N=n_particles, xstar=self.lambdas[:, i, 0])
            pf.run()
            # Extraction trajectoire T+1 -> T
            full_traj = np.array(pf.hist.extract_one_trajectory())
            self.lambdas[:, i, 0] = full_traj[-self.T:].flatten()
        self.first_iter = False

    def update_nu(self, rw_step=0.15):
        """Étape 3 : Metropolis-Hastings pour nu sur la FULL COPULA."""
        if not self.is_student: return
        prop_nu = self.nu + np.random.normal(0, rw_step)
        if prop_nu <= 2.1 or prop_nu > 40.0: return
        
        ll_old = self._compute_full_log_lik(self.nu, self.x_data)
        x_prop = student_t.ppf(self.u_data, df=prop_nu)
        ll_new = self._compute_full_log_lik(prop_nu, x_prop)
        
        log_prior_old = stats.gamma.logpdf(self.nu - 2, a=2.5, scale=2.0)
        log_prior_new = stats.gamma.logpdf(prop_nu - 2, a=2.5, scale=2.0)
        
        if np.log(np.random.rand()) < (ll_new + log_prior_new - ll_old - log_prior_old):
            self.nu = prop_nu
            self.x_data = x_prop

    def update_params_ar1(self):
        """Étape 4 : Paramètres AR(1)."""
        T_eff = self.T - 1
        for i in range(self.n):
            for l in range(self.p):
                lam_t, lam_next = self.lambdas[:-1, i, l], self.lambdas[1:, i, l]
                phi, mu = self.theta['phi'][i, l], self.theta['mu'][i, l]
                sse = np.sum((lam_next - (mu + phi * (lam_t - mu)))**2)
                sig_sq = 1.0 / np.random.gamma(20.0 + T_eff/2.0, 1.0 / (0.25 + sse/2.0))
                self.theta['sigma'][i, l] = np.sqrt(sig_sq)
                prec_mu = (1.0/2.0) + (T_eff * (1-phi)**2 / sig_sq)
                mean_mu = ((0.4/2.0) + (1-phi) * np.sum(lam_next - phi*lam_t) / sig_sq) / prec_mu
                self.theta['mu'][i, l] = np.random.normal(mean_mu, 1.0/np.sqrt(prec_mu))
                Xc, Yc = lam_t - self.theta['mu'][i, l], lam_next - self.theta['mu'][i, l]
                prec_phi = (1.0/0.001) + (np.sum(Xc**2) / sig_sq)
                mean_phi = ((0.985/0.001) + np.sum(Xc*Yc) / sig_sq) / prec_phi
                std_phi = 1.0/np.sqrt(prec_phi)
                a, b = (-1.0 - mean_phi)/std_phi, (1.0 - mean_phi)/std_phi
                self.theta['phi'][i, l] = stats.truncnorm.rvs(a, b, loc=mean_phi, scale=std_phi)

    def _compute_full_log_lik(self, nu_val, x_val):
        """Vraisemblance totale du dataset."""
        total_ll = 0.0
        for t in range(self.T):
            # Construction locale simplifiée de R_t pour la vitesse
            lam = self.lambdas[t]
            lam_sq = np.sum(lam**2, axis=1)
            den = np.sqrt(np.outer(1+lam_sq, 1+lam_sq))
            R = (lam @ lam.T) / den
            np.fill_diagonal(R, 1.0)
            det_R = np.linalg.det(R)
            if det_R <= 1e-10: continue
            R_inv = np.linalg.inv(R)
            quad = x_val[t] @ R_inv @ x_val[t]
            log_joint = (special.gammaln((nu_val + self.n) / 2.0) - special.gammaln(nu_val / 2.0) 
                        - (self.n / 2.0) * np.log(nu_val * np.pi) - 0.5 * np.log(det_R)
                        - ((nu_val + self.n) / 2.0) * np.log(1.0 + quad / nu_val))
            log_marg = np.sum(special.gammaln((nu_val + 1) / 2.0) - special.gammaln(nu_val / 2.0)
                             - 0.5 * np.log(nu_val * np.pi) - ((nu_val + 1) / 2.0) * np.log(1.0 + x_val[t]**2 / nu_val))
            total_ll += (log_joint - log_marg)
        return total_ll

    def run(self, n_iter, n_particles=100):
        for _ in tqdm(range(n_iter)):
            self.update_z()
            self.update_lambdas_pg(n_particles)
            if self.is_student: self.update_nu()
            self.update_params_ar1()
            self.trace['nu'].append(self.nu)
            self.trace['phi'].append(np.mean(self.theta['phi']))