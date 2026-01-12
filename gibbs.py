import numpy as np
from scipy import stats, special, linalg
import particles
from particles import state_space_models as ssm
from particles import distributions
from particles import mcmc 
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm

class IdentifiedLoadingSSM(ssm.StateSpaceModel):
    """
    State space model à utilisé pour SMCS dans le Gibbs lorsque l'on suppose les z et zeta connus, avec phi et sigma diagnale state space sont idépendants.
    """
    def __init__(self, row_idx, p_factors, mu, phi, sigma_eta, z, zeta, data_x):
        self.i = row_idx
        self.p = p_factors

        # Dimension effective
        self.dim_state = min(row_idx + 1, p_factors)
        
        # Paramètres
        self.mu = mu[:self.dim_state]
        self.phi = phi[:self.dim_state]
        self.sigma_eta = sigma_eta[:self.dim_state]
        
        self.z = z       
        self.zeta = zeta 
        self.data_x = data_x 
    
    def PX0(self):
        """Distribution initiale avec variance gonflée (Diffuse Prior)"""       
        # L'article semble suggére
        var_0 = self.sigma_eta**2 * 100.0
        
        cov_0 = np.diag(var_0)
        return distributions.MvNormal(loc=self.mu, cov=cov_0)

    def PX(self, t, xp):
        """Transition : MvNormal"""
        # xp est (N, dim_state). mean sera (N, dim_state)
        mean = self.mu + self.phi * (xp - self.mu)
        cov_eta = np.diag(self.sigma_eta**2)
        
        return distributions.MvNormal(loc=mean, cov=cov_eta)

    def PY(self, t, xp, x):
        """Vraisemblance"""
        N = x.shape[0]
        
        # 1. Reconstitution
        real_loadings = np.zeros((N, self.p)) 
        real_loadings[:, :self.dim_state] = x
        
        # 2. Transformation Log -> Exp
        if self.i < self.p:
            real_loadings[:, self.i] = np.exp(x[:, self.i])
            
        # 3. Paramètres
        z_t = self.z[t]
        factor_comp = np.dot(real_loadings, z_t)
        
        lam_sq = np.sum(real_loadings**2, axis=1)
        scaling = np.sqrt(1.0 + lam_sq)
        sigma_eps = 1.0 / scaling
        
        zeta_sqrt = np.sqrt(self.zeta[t])
        
        mean_obs = zeta_sqrt * (factor_comp / scaling)
        scale_obs = zeta_sqrt * sigma_eps
        
        return distributions.Normal(loc=mean_obs, scale=scale_obs)
    
def process_single_asset(i, p, mu_i, phi_i, sigma_i, z, zeta_i, data_i, first_iter, x_prev_i=None):
        """Traite un seul actif : Crée le SSM, lance le SMC/CSMC et retourne la trajectoire. Utile pour joblib."""
        
        ssm_i = IdentifiedLoadingSSM(
            row_idx=i, p_factors=p,
            mu=mu_i, phi=phi_i, sigma_eta=sigma_i,
            z=z, zeta=zeta_i, data_x=data_i
        )
        
        if first_iter:
            cpf = particles.SMC(fk=ssm.Bootstrap(ssm=ssm_i, data=data_i), 
                            N=100, store_history=True)
        else:
            cpf = mcmc.CSMC(fk=ssm.Bootstrap(ssm=ssm_i, data=data_i), 
                            N=100, xstar=x_prev_i)
        
        cpf.run()
        
        raw_traj_list = cpf.hist.extract_one_trajectory()
        traj = np.array(raw_traj_list)
        
        T = data_i.shape[0]
        dim = ssm_i.dim_state
        
        if traj.ndim > 2: traj = np.squeeze(traj)
        if traj.ndim == 1: traj = traj[:, np.newaxis]
        if traj.shape[0] == T + 1: traj = traj[1:]
        if traj.shape != (T, dim): traj = traj.reshape(T, dim)
            
        lambdas_vals = np.zeros((T, p))
        lambdas_vals[:, :dim] = traj
        if i < p:
            lambdas_vals[:, i] = np.exp(traj[:, i])
            
        return i, traj, lambdas_vals
    

class GibbsFactorCopula:
    "On reprend les étapes dans le même ordre que l'appendix de l'article"
    def __init__(self, data, n_factors, nu = 10):
        self.data = data
        self.T, self.n = data.shape
        self.p = n_factors
        
        self.theta = {
            'mu': np.ones((self.n, self.p)) * 0.5,
            'phi': np.ones((self.n, self.p))* 0.85,
            'sigma': np.ones((self.n, self.p)) * 0.1
        }
        
        self.z = np.random.normal(size=(self.T, self.p))
        self.zeta = np.ones((self.T, self.n))
        self.nu = nu
        
        self.latent_states = np.ones((self.T, self.n, self.p))*0.5
        self.lambdas = np.ones((self.T, self.n, self.p))*0.5
        
        for i in range(self.p):
            self.lambdas[:, i, i] = 0.1
            self.latent_states[:, i, i] = np.log(0.1)
            
        self.first_iter = True

    def update_zeta(self):
        "Forme ferme pour G=1 avec zeta_t vecteur (N,)"
        T, N = self.T, self.n
        alpha_post = (self.nu + N) / 2.0
        
        for t in range(T):
            lam_t = self.lambdas[t] 
            S_sq = 1.0 + np.sum(lam_t**2, axis=1)
            inv_S_sq = 1.0 / S_sq
            
            x_scaled = self.data[t] * np.sqrt(S_sq)

            resid = x_scaled - np.dot(lam_t, self.z[t])
            d_t_sq = np.sum(resid**2) 
            

            beta_post = (self.nu + d_t_sq) / 2.0
            self.zeta[t, :] = 1.0 / np.random.gamma(alpha_post, 1.0 / beta_post)


    def update_zeta_old(self):
        """
        Step: Mise à jour des variables de mélange zeta_t (Cas G=1).
        On utilise la propriété : Si X ~ Gamma(a, b), alors 1/X ~ InvGamma(a, b).
        """
        T, N = self.T, self.n
        
        alpha_post = (self.nu + N) / 2.0
        

        new_zetas = np.zeros((T, 1))
        
        for t in range(T):
            x_t = self.data[t]       
            z_t = self.z[t]          
            lam_t = self.lambdas[t]  
            
            mean_t = np.dot(lam_t, z_t)
            
            resid = x_t - mean_t
            
            d_t = np.sum(resid**2)

            beta_post = (self.nu + d_t) / 2.0

            
            rho_val = np.random.gamma(shape=alpha_post, scale=(1.0 / beta_post))
            
            new_zetas[t] = 1.0 / rho_val
            
        self.zeta = np.tile(new_zetas, (1, N))

    def log_prior_nu(self, nu):
        """
        Calcule le log-prior de nu basé
        """
        # Contrainte stricte : nu > 2 (sinon variance infinie)
        if nu <= 2.0:
            return -np.inf
            
        # Variable transformée
        nu_tilde = nu - 2.0
        return stats.gamma.logpdf(nu_tilde, a=2.5, scale=2.0)

    def log_pdf_ig_nu(self, zetas, nu):
        """
        Calcule la log-vraisemblance de nu sachant les zeta.
        zeta_t ~ InvGamma(nu/2, nu/2)
        """
        T = len(zetas)
        half_nu = nu / 2.0
        
        term1 = T * (half_nu * np.log(half_nu) - special.gammaln(half_nu))
        
        sum_log_zeta = np.sum(np.log(zetas))
        sum_inv_zeta = np.sum(1.0 / zetas)
        
        term2 = - (half_nu + 1) * sum_log_zeta - half_nu * sum_inv_zeta
        
        return term1 + term2

    def update_nu(self, rw_step=0.5):
        """
        Step: Mise à jour de nu avec Metropolis-Hastings. Diférent de l'article qui intègre le fait que
         la transfo u -> x depend de nu
        """
        current_nu = self.nu
        
        zetas = self.zeta[:, 0]
        proposal_nu = current_nu + np.random.normal(0, rw_step)
        
        if proposal_nu <= 2.0:
            self.nu = current_nu
            return

        log_lik_new = self.log_pdf_ig_nu(zetas, proposal_nu)
        log_lik_old = self.log_pdf_ig_nu(zetas, current_nu)
        

        log_prior_new = self.log_prior_nu(proposal_nu)
        log_prior_old = self.log_prior_nu(current_nu)
        
        acceptance_log_prob = (log_lik_new + log_prior_new) - (log_lik_old + log_prior_old)
        
        # Acceptation
        if np.log(np.random.rand()) < acceptance_log_prob:
            self.nu = proposal_nu
        # Sinon on garde self.nu inchangé


    def update_z(self):
        """Inspiré de regression bayésienne multivariée."""
        T, n, p = self.T, self.n, self.p
        new_z = np.zeros((T, p))
        I_p = np.eye(p)
        
        for t in range(T):
            x_t = self.data[t]
            lambda_t = self.lambdas[t]
            zeta_t = self.zeta[t]
            
            lam_sq = np.sum(lambda_t**2, axis=1)
            sigma_sq = 1.0 / (1.0 + lam_sq)
            sigma = np.sqrt(sigma_sq)
            
            lambda_tilde = lambda_t * sigma[:, np.newaxis]
            x_dot = x_t / np.sqrt(zeta_t)
            
            D_inv_diag = 1.0 / sigma_sq
            Ct_Dinv = lambda_tilde.T * D_inv_diag[np.newaxis, :]
            
            prec_post = I_p + Ct_Dinv @ lambda_tilde
            linear_term = Ct_Dinv @ x_dot
            
            try:
                L = linalg.cholesky(prec_post, lower=True)
                mu_post = linalg.cho_solve((L, True), linear_term)
            except linalg.LinAlgError:
                L = linalg.cholesky(prec_post + 1e-6*np.eye(p), lower=True)
                mu_post = linalg.cho_solve((L, True), linear_term)
                
            epsilon = np.random.normal(size=p)
            z_noise = linalg.solve_triangular(L.T, epsilon, lower=False)
            new_z[t] = mu_post + z_noise
            
        self.z = new_z

    
    def update_lambda_pg_parallel(self, n_jobs=-1):
        """Version parallélisée de update_lambda_pg"""
        T, n = self.data.shape
        
        tasks = []
        for i in range(n):
            x_prev = None
            if not self.first_iter:
                dim = min(i + 1, self.p)
                x_prev_numpy = self.latent_states[:, i, :dim]
                x_prev = [x_prev_numpy[t] for t in range(T)]
            
            tasks.append((
                i, self.p, 
                self.theta['mu'][i], self.theta['phi'][i], self.theta['sigma'][i],
                self.z, self.zeta[:, i], self.data[:, i], 
                self.first_iter, x_prev
            ))
            
        # Exécution parallèle (n_jobs=-1 utilise tous les cœurs)
        results = Parallel(n_jobs=n_jobs)(delayed(process_single_asset)(*t) for t in tasks)
        
        for i, traj, lam_vals in results:
            dim = min(i + 1, self.p)
            self.latent_states[:, i, :dim] = traj
            self.lambdas[:, i, :] = lam_vals
            
        self.first_iter = False



    def update_lambda_pg(self):
        """
        Step 5: Particle Gibbs pour Lambda.
        On utilise CSMC.
        """
        T, n = self.data.shape
        p = self.p
        
        new_lambdas = np.zeros((T, n, p))
        new_states = np.zeros((T, n, p))
        
        mu_vec = self.theta['mu']
        phi_vec = self.theta['phi']
        sigma_vec = self.theta['sigma']
        
        for i in range(n):
            data_i = self.data[:, i]
            
            ssm_i = IdentifiedLoadingSSM(
                row_idx=i, p_factors=p,
                mu=mu_vec[i], phi=phi_vec[i], sigma_eta=sigma_vec[i],
                z=self.z, zeta=self.zeta[:, i], data_x=data_i
            )
            
            if self.first_iter:
                cpf = particles.SMC(fk=ssm.Bootstrap(ssm=ssm_i, data=data_i), 
                                  N=100, store_history=True)
            else:
                dim = ssm_i.dim_state
                x_prev_numpy = self.latent_states[:, i, :dim]

                x_prev = [x_prev_numpy[t] for t in range(T)]
                
                cpf = mcmc.CSMC(fk=ssm.Bootstrap(ssm=ssm_i, data=data_i), 
                                N=100, xstar=x_prev)
            
            cpf.run()
            

            raw_traj_list = cpf.hist.extract_one_trajectory()
            
            traj = np.array(raw_traj_list)
            
            dim = ssm_i.dim_state
            
            if traj.ndim > 2:
                traj = np.squeeze(traj)
            if traj.ndim == 1:
                traj = traj[:, np.newaxis]
                
            if traj.shape[0] == T + 1:
                traj = traj[1:]
            
            if traj.shape != (T, dim):
                traj = traj.reshape(T, dim)

            new_states[:, i, :dim] = traj
            new_lambdas[:, i, :dim] = traj
            
            if i < p:
                new_lambdas[:, i, i] = np.exp(traj[:, i])
                new_lambdas[:, i, dim:] = 0.0
                
        self.lambdas = new_lambdas
        self.latent_states = new_states
        self.first_iter = False

    def update_sigma(self):
        """Step 6: Inverse Gamma"""
        h_curr = self.latent_states[:-1]
        h_next = self.latent_states[1:]
        
        mu = self.theta['mu'][np.newaxis, :, :]
        phi = self.theta['phi'][np.newaxis, :, :]
        
        pred = mu + phi * (h_curr - mu)
        resid = h_next - pred
        sse = np.sum(resid**2, axis=0) 
        
        alpha_post = 20.0 + (h_curr.shape[0] / 2.0)
        beta_post = 0.25 + (sse / 2.0)
        
        self.theta['sigma'] = 1.0 / np.random.gamma(alpha_post, 1.0/beta_post)
    
    def update_mu_phi(self):
        h_curr = self.latent_states[:-1] # (T-1, N, P)
        h_next = self.latent_states[1:]  # (T-1, N, P)
        T_eff = h_curr.shape[0]
        sigma_sq = self.theta['sigma']   # (N, P)
        phi = self.theta['phi']           # (N, P)
        

        X_mu = 1.0 - phi
        Y_mu = h_next - phi * h_curr 
        sum_Y = np.sum(Y_mu, axis=0)
        
        prec_prior = 1.0 / 2.0 
        prec_lik = T_eff * (X_mu**2) / sigma_sq
        prec_post = prec_prior + prec_lik
        
        num_post = (prec_prior * 0.4) + (X_mu / sigma_sq) * sum_Y
        mean_post = num_post / prec_post
        
        self.theta['mu'] = np.random.normal(mean_post, np.sqrt(1.0/prec_post))
        
        mu = self.theta['mu']
        Xc = h_curr - mu 
        Yc = h_next - mu 
        
        num_phi = np.sum(Xc * Yc, axis=0)
        den_phi = np.sum(Xc**2, axis=0)
        
        prec_prior_phi = 1.0 / 0.1
        prec_lik_phi = den_phi / sigma_sq
        prec_post_phi = prec_prior_phi + prec_lik_phi
        
        mean_post_phi = (prec_prior_phi * 0.985 + num_phi / sigma_sq) / prec_post_phi
        std_post_phi = np.sqrt(1.0/prec_post_phi)
        
        a = (-1.0 - mean_post_phi) / std_post_phi
        b = (1.0 - mean_post_phi) / std_post_phi
        self.theta['phi'] = stats.truncnorm.rvs(a, b, loc=mean_post_phi, scale=std_post_phi)

    def update_mu_phi_old(self):
        h_curr = self.latent_states[:-1]
        h_next = self.latent_states[1:]
        T_eff = h_curr.shape[0]
        sigma_sq = self.theta['sigma']
        
        phi = self.theta['phi']
        Y = h_next - phi[np.newaxis, :, :] * h_curr
        X = 1.0 - phi
        
        prec_prior = 1.0 / 2.0 
        prec_lik = T_eff * (X**2) / sigma_sq
        prec_post = prec_prior + prec_lik
        
        mean_lik = (X / sigma_sq) * np.sum(Y, axis=0)
        mean_post = (1.0/prec_post) * (prec_prior*0.4 + mean_lik)
        
        self.theta['mu'] = np.random.normal(mean_post, np.sqrt(1.0/prec_post))
        
        mu = self.theta['mu']
        X_c = h_curr - mu[np.newaxis, :, :]
        Y_c = h_next - mu[np.newaxis, :, :]
        
        num = np.sum(X_c * Y_c, axis=0)
        den = np.sum(X_c**2, axis=0)
        
        prec_prior_phi = 1.0 / 0.001
        prec_lik_phi = den / sigma_sq
        prec_post_phi = prec_prior_phi + prec_lik_phi
        
        mean_post_phi = (1.0/prec_post_phi) * (prec_prior_phi*0.985 + (num/sigma_sq))
        std_post_phi = np.sqrt(1.0/prec_post_phi)
        
        a = (-1.0 - mean_post_phi) / std_post_phi
        b = (1.0 - mean_post_phi) / std_post_phi
        self.theta['phi'] = stats.truncnorm.rvs(a, b, loc=mean_post_phi, scale=std_post_phi)
