# Large Scale Variational Smoothing
Approximate inference targeted at variational Gaussian state-space models with dense covariance matrix approximations.  For more details, see our paper: [Dowling, Zhao, Park. 2024](https://arxiv.org/abs/2403.01371)


### introduction
The core of a large scale variational smoothing (LSVS) module is a LowRankNonlinearStateSpaceModel object.  A LowRankNonlinearStateSpaceModel is used to perform inference in a state-space graphical model specified by,

$$p(y_{1:T}, z_{1:T}) = p_{\theta}(z_1) p(y_1 | z_1) \prod p_{\psi}(y_t | z_t) p_{\theta}(z_t | z_{t-1})$$
where
$$p_{\theta}(z_1) = N(m_0, Q_0)$$
$$p_{\theta}(z_t | z_{t-1}) = N(m_{\theta}(z_{t-1}), Q)$$

Specification of a LowRankNonlinearStateSpaceModel requires:

1. dynamics_mod: a nn.Module with a `mean_fn` attribute that specifies $m_{\theta}(\cdot)$ (e.g. some neural network function) 
2. initial_c_pdf: a nn.Module that specifies the initial mean, $m_0$, and initial state-noise $Q_0$
3. likelihood_pdf: a nn.Module that implements a `get_ell(y, z)` method that returns $\log p_{\psi}(y|z)$
4. local_encoder: a nn.Module whose `forward` function maps time-instantaneous observations, $y_t$, to a natural parameter update, $\alpha_t$
5. backward_encoder: a nn.Module whose `forward` function maps sequences of local natural parameter updates, $\alpha_{1:T}$, to backward natural parameter updates, $\beta_{1:T}$
6. nl_filter: a nn.Module whose forward method takes natural parameter representation of observations, $\tilde{\lambda}_{1:T}$, and performs approximate posterior inference for the specified nonlinear dynamical system

### parameter descriptions
There are several configuration settings that can be configured to trade off expressivity/speed/generalization properties of the model and inference algorithm that we detail below.  Some are specific to the case of neural data modeled using a Poisson observation model with log-linear readout of the latent state.

1. n_latents: (int) total number of latent variables
2. n_latents_read: (int) total number of latent variables accessed by the observation model
3. rank_local: (int) the rank of the local precision update -- $A(y_t) A(y_t)^{\top}$
4. rank_backward: (int) the rank of the backward preecision update -- $B(\alpha_{t+1:T}) B(\alpha_{t+1:T})^{\top}$
5. p_mask_a: (float) the probability of setting $\alpha_t$ to $0$
6. p_mask_b: (float) the probability of setting $\beta_t$ to $0$
7. p_mask_apb: (float) the probability of setting $\tilde{\lambda}_t = \alpha_t + \beta_t$ to $0$
8. p_mask_y_in: (float) the probability of masking data fed as input to the inference networks
9. use_cd: (bool) whether to use the coordinated dropout technique of [Keshtkaran and Pandarinath, 2019](https://arxiv.org/abs/1908.07896) -- in which case `p_mask_y_in` specifies the coordinated dropout rate

Setting `p_mask_a` is equivalent to masking *actual* observations, $y_t$; this strategy was used in the context of structured VAE's for linear dynamical systems in [Zhao, and Linderman. 2023](https://arxiv.org/abs/2305.16543) to promote learning dynamics more adept at prediction (and thus generating more realistic data).  Setting `p_mask_b` is equivalent to masking `pseudo` observations, $\tilde{y}_t$ -- this helps to regularize both the local/backward encoders required.

### example configuration
LSVS was designed with custom configurations in mind so that depending on the problem, `dynamics_mod`, `initial_c_pdf`, `likelihood_pdf`, `local_encoder`, and `backward_encoder` can be configured as desired.  We include some general classes in `ssm_modules/encoders`, `ssm_modules/likelihoods` and `ssm_modules/dynamics` that should be sufficient for a wide range of problems.  Below is an example configuration.
```
    """likelihood pdf"""
    C = torch.nn.Linear(cfg.n_latents, n_neurons_obs, device=cfg.device)
    likelihood_pdf = PoissonLikelihood(C, n_neurons_obs, cfg.bin_sz, device=cfg.device)

    """dynamics module"""
    Q = torch.ones(cfg.n_latents, device=cfg.device)
    dynamics_fn = utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device)
    dynamics_mod = DenseGaussianNonlinearDynamics(dynamics_fn, cfg.n_latents, Q, device=cfg.device)

    """initial condition"""
    Q0 = torch.ones(cfg.n_latents, device=cfg.device)
    m0 = torch.zeros(cfg.n_latents, device=cfg.device)
    initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m0, Q0, device=cfg.device)

    """local/backward encoder"""
    backward_encoder = BackwardEncoderLRMvn(cfg.n_latents_read, cfg.n_hidden_backward, cfg.n_latents, cfg.rank_local,
                                            cfg.rank_backward, device=cfg.device)
    local_encoder = LocalEncoderLRMvn(cfg.n_latents, n_neurons_enc, cfg.n_hidden_local, cfg.n_latents,
                                      cfg.rank_local, device=cfg.device, dropout=cfg.p_local_dropout)
    nl_filter = NonlinearFilter(dynamics_mod, initial_condition_pdf, device=cfg.device)

    """sequence vae"""
    ssm = LowRankNonlinearStateSpaceModel(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder, local_encoder, nl_filter, device=cfg.device)


    """lightning"""
    seq_vae = LightningNlbNonlinearSSM(ssm, cfg)

```


### acknowledgements
Structure of the code and configuration management was heavily inspired by the excellently written `lfads-torch` package at [https://github.com/arsedler9/lfads-torch](https://github.com/arsedler9/lfads-torch) as described in [Sedler and Pandarinath, 2023](https://arxiv.org/abs/2309.01230).