"""A computable example of the theory of Sections 3-5: a scalar linear system with Gaussian noise (Appendix D).

Code for Appendix D of "Dissimilarity-Based Markov Decision Processes for Stabilizing State Distributions".
Every hypothesis of Theorems 2-5 can be checked in this example, and every quantity of their certificates computed.

Modules
-------
system     dynamics, exact Gaussian closure f, target, Omega_com, D = W_2^2, estimators   (Appendix D, eq. parameter_dynamics)
grids      graded and uniform grids on Omega_com and A, bilinear interpolation           (Appendix D)
vi         ratio value iteration for (P-OCP) and (AP-OCP^R), greedy policy, error bounds  (Theorem 2, Section 4.2)
gp         SE-ARD Gaussian-process regression with ML-II hyperparameters                (Appendix C)
offline    offline data, learned model, learned value and policy, quantities of Theorem 3 (Section 5.1)
online     sample-size schedule, plug-in perturbation, plug-in closed loops              (Theorems 4 and 5)
modelfree  model-free learner with W_n = D, quantities of Theorem 5                     (Section 5.2)
certify    test sets, order-statistic calibration, hypothesis table                     (Theorems 3-5, Proposition 1)
plots      figure fig:toy_certificates and the tables tab:toy_hypotheses, tab:toy_results (Appendix D)
stages     the resumable stages true, offline, online, modelfree, outputs               (run_toy.py)
"""
