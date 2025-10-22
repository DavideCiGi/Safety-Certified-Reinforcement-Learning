import sympy as sp
import numpy as np
import cvxpy as cp
import math
from typing import Sequence, Tuple, List, Dict
from utils import HistoryTracker
from buffer import GaussianProcessBuffer


def auxs_fun_expr(h: Sequence[sp.Expr], gamma: Sequence[float]) -> Tuple[sp.Expr, sp.Expr]:
    """
    Compute the r-th auxiliary function expression.

    Parameters
    ----------
    h : Sequence[sympy.Expr]
        Sequence of SymPy expressions.
    gamma : Sequence[float]
        Sequence of numeric coefficients.

    Returns
    -------
    Tuple[sympy.Expr, sym.expr]
        A tuple with the resulting SymPy expressions.

    Raises
    ------
    ValueError
        If len(h) - 1 != len(gamma).
    """
    r = len(h) - 1
    if r != len(gamma):
        raise RuntimeError('Check the number of auxiliary functions parameters!')
    psi = sp.Matrix(r+1, r+1, lambda i, j: sp.symbols(f'psi_{i}{j}', real=True))
    for j in range(r + 1):
        psi[0, j] = h[j]
    for i in range(r):
        for j in range(r - i):
            psi[i + 1, j] = psi[i, j + 1] - psi[i, j] + gamma[i] * psi[i, j]
    psi_r = psi[-1, 0]
    psi_no_r = psi[:-1, 0]
    return psi_r, psi_no_r


def bergman_t1d_sym_dynamics(dt: float, params: Dict[str, float],  step: int) -> (
        Tuple)[List[sp.Symbol], List[sp.Symbol], int, int]:
    """
    Generate symbolic variables for the discrete-time Bergman T1D dynamics.

    Parameters
    ----------
    dt : float
        Sampling time.
    params: Dict[str, float]
        Values of the parameters of the model
    step : int
        Step up to which you want to calculate the symbolic state dynamics.

    Returns
    -------
    Tuple[List[sp.Symbol], List[sp.Symbol], int, int]
        A tuple consisting in two symbolic lists, a state and an input one consisting, respectively,
        of the state symbolic variables calculated up to "step" and of the input symbolic variables up to "step". The
        other elements of the tuple, the two integers, are respectively the state and input dimension.
    """

    p_1, G_b, p_2, p_3, n, I_b = params['p_1'], params['G_b'], params['p_2'], params['p_3'], params['n'], params['I_b']
    tau_G, V_G = params['tau_G'], params['V_G']

    G = [sp.symbols(f"G({i})", real=True) for i in range(step + 1)]
    X = [sp.symbols(f"X({i})", real=True) for i in range(step + 1)]
    I = [sp.symbols(f"I({i})", real=True) for i in range(step + 1)]
    D_2 = [sp.symbols(f"D_{{2}}({i})", real=True) for i in range(step + 1)]
    D_1 = [sp.symbols(f"D_{{1}}({i})", real=True) for i in range(step + 1)]

    ID = [sp.symbols(f"ID({i})", real=True) for i in range(step + 1)]
    CHO = [sp.symbols(f"CHO({i})", real=True) for i in range(step + 1)]

    for i in range(step):
        G[i + 1] = G[i] + dt * (- p_1 * (G[i] - G_b) - G[i] * X[i] + (1 / (V_G * tau_G)) * D_2[i])
        X[i + 1] = X[i] + dt * (- p_2 * X[i] + p_3 * (I[i] - I_b))
        I[i + 1] = I[i] + dt * (- n * (I[i] - I_b) + ID[i])
        D_2[i + 1] = D_2[i] + dt * (- D_2[i] / tau_G + D_1[i] / tau_G)
        D_1[i + 1] = D_1[i] + dt * (- D_1[i] / tau_G + CHO[i])

    sym_state = G + X + I + D_2 + D_1
    sym_input = ID + CHO

    state_dim = len(sym_state) // (step + 1)
    input_dim = len(sym_input) // (step + 1)

    return sym_state, sym_input, state_dim, input_dim


class ViolatedBergmanT1DDiscreteTimeHighOrderControlBarrierFunctions:
    def __init__(self, gamma_1, gamma_2):
        self.BG_min = 90.
        self.BG_max = 180.
        self.r = 3

        G_to_subs = [sp.symbols(f"G({i})", real=True) for i in range(self.r + 1)]
        X_to_subs = [sp.symbols(f"X({i})", real=True) for i in range(self.r + 1)]
        I_to_subs = [sp.symbols(f"I({i})", real=True) for i in range(self.r + 1)]
        D_2_to_subs = [sp.symbols(f"D_{{2}}({i})", real=True) for i in range(self.r + 1)]
        D_1_to_subs = [sp.symbols(f"D_{{1}}({i})", real=True) for i in range(self.r + 1)]

        self.epsilon1_to_sub = sp.symbols(r"\epsilon_1", real=True)
        self.epsilon2_to_sub = sp.symbols(r"\epsilon_2", real=True)

        # ... define the violated HODTCBFs...
        h1 = [g - self.BG_min + self.epsilon1_to_sub for g in G_to_subs]
        h2 = [- g + self.BG_max + self.epsilon2_to_sub for g in G_to_subs]

        self.sym_state_to_subs = G_to_subs + X_to_subs + I_to_subs + D_2_to_subs + D_1_to_subs
        self.psi1_r_violated, self.psi1_no_r_violated = auxs_fun_expr(h1, gamma_1)
        self.psi2_r_violated, self.psi2_no_r_violated = auxs_fun_expr(h2, gamma_2)
        self.psi1_r = self.psi1_r_violated.subs({self.epsilon1_to_sub: 0.}).copy()
        self.psi1_no_r = self.psi1_no_r_violated.subs({self.epsilon1_to_sub: 0.}).copy()
        self.psi2_r = self.psi2_r_violated.subs({self.epsilon2_to_sub: 0.}).copy()
        self.psi2_no_r = self.psi2_no_r_violated.subs({self.epsilon2_to_sub: 0.}).copy()

        self.state_dim = len(self.sym_state_to_subs) // (self.r + 1)
        self.init_X_ht = HistoryTracker(self.r - 1, 1, self.state_dim)
        self.collect_cntr = 0

    def collect(self, state):
        self.init_X_ht.add(state)
        self.collect_cntr += 1

    def safe_set1_check(self, epsilon1=None):
        if self.collect_cntr <= self.r - 1:
            return None

        if epsilon1 is None:
            epsilon1 = 0.

        psi1_init = self.psi1_no_r_violated.subs({self.epsilon1_to_sub: epsilon1}).copy()
        k1 = 0
        for i in range(self.state_dim):
            for j in range(self.r):
                for k in range(j + 1):
                    psi1_init[j] = (psi1_init[j]).subs({self.sym_state_to_subs[k + i * (self.r + 1)]
                                                        : self.init_X_ht.history[-(k + 1), i]})
        for j in range(self.r):
            if psi1_init[j] >= 0:
                k1 += 1

        return k1 == self.r

    def safe_set2_check(self, epsilon2=None):
        if self.collect_cntr < self.r - 1:
            return None
        if epsilon2 is None:
            epsilon2 = 0.

        psi2_init = self.psi2_no_r_violated.subs({self.epsilon2_to_sub: epsilon2}).copy()
        k2 = 0
        for i in range(self.state_dim):
            for j in range(self.r):
                for k in range(j + 1):
                    psi2_init[j] = (psi2_init[j]).subs({self.sym_state_to_subs[k + i * (self.r + 1)]
                                                        : self.init_X_ht.history[-(k + 1), i]})
        for j in range(self.r):
            if psi2_init[j] >= 0:
                k2 += 1

        return k2 == self.r

    def reset_collect(self):
        self.init_X_ht = HistoryTracker(self.r - 1, 1, self.state_dim)
        self.collect_cntr = 0

    def calculate(self, dt, params):  # IF YOU CHANGE THE SYSTEM DO NOT TOUCH THE LOGIC OF THIS METHOD
        sym_state, sym_all_input, state_dim, input_dim = bergman_t1d_sym_dynamics(dt, params, self.r)
        # you can implement a check of the relative degree here! Using sym_input from bergman_t1d_sym_dynamics

        psi1_r_eval = self.psi1_r.subs(dict(zip(self.sym_state_to_subs, sym_state)))
        psi2_r_eval = self.psi2_r.subs(dict(zip(self.sym_state_to_subs, sym_state)))

        if (set([sym_all_input[k*(self.r+1)] for k in range(input_dim)]) !=
                psi1_r_eval.free_symbols.intersection(set(sym_all_input))):
            raise RuntimeError(f'The HODTCBFs are not of order {self.r} for the {input_dim} input!')

        sym_actual_input = [sym_all_input[k*(self.r+1)] for k in range(input_dim)]
        sym_init_state = [sym_state[k*(self.r+1)] for k in range(state_dim)]

        return psi1_r_eval, psi2_r_eval, sym_init_state, sym_actual_input, state_dim, input_dim


class BergmanT1DDTHOCBFsUtils(ViolatedBergmanT1DDiscreteTimeHighOrderControlBarrierFunctions):
    def __init__(self, gamma_1, gamma_2, dt, CHO_max, nominal_params, true_params, GP_max_size, episode_length,
                 GP_collect_sigma_1=None, GP_collect_sigma_2=None):
        super().__init__(gamma_1, gamma_2)
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        psi1_r_eval, psi2_r_eval, sym_init_state, (sym_ID, sym_CHO), self.state_dim, self.action_dim =\
            self.calculate(dt, nominal_params)
        no_meal_psi1_r_eval = psi1_r_eval.subs({sym_CHO: 0.})
        no_meal_psi2_r_eval = psi2_r_eval.subs({sym_CHO: 0.})
        robust_psi1_r_eval = psi1_r_eval.subs({sym_CHO: 0.})
        robust_psi2_r_eval = psi2_r_eval.subs({sym_CHO: CHO_max})
        true_psi1_r_eval, true_psi2_r_eval, *_ = self.calculate(dt, true_params)
        robust_true_psi1_r_eval = true_psi1_r_eval.subs({sym_CHO: 0.})
        robust_true_psi2_r_eval = true_psi2_r_eval.subs({sym_CHO: CHO_max})

        sym_init_state_control = sym_init_state + [sym_ID]

        self.no_meal_psi1_r_fun = sp.lambdify(sym_init_state_control, no_meal_psi1_r_eval, modules='numpy')
        self.no_meal_psi2_r_fun = sp.lambdify(sym_init_state_control, no_meal_psi2_r_eval, modules='numpy')
        self.robust_psi1_r_fun = sp.lambdify(sym_init_state_control, robust_psi1_r_eval, modules='numpy')
        self.robust_psi2_r_fun = sp.lambdify(sym_init_state_control, robust_psi2_r_eval, modules='numpy')
        self.robust_true_psi1_r_fun = sp.lambdify(sym_init_state_control, robust_true_psi1_r_eval, modules='numpy')
        self.robust_true_psi2_r_fun = sp.lambdify(sym_init_state_control, robust_true_psi2_r_eval, modules='numpy')

        # self.action_dim is the dimension of ALL inputs. The controllable ones are of dimension self.action_dim - 1
        self.GP_memory = GaussianProcessBuffer(GP_max_size, self.state_dim, self.action_dim - 1, episode_length)
        self.X_ht = HistoryTracker(self.r, 1, self.state_dim + self.action_dim - 1)

        if GP_collect_sigma_1 is None:
            self.GP_collect_sigma_1 = 0.
        else:
            self.GP_collect_sigma_1 = GP_collect_sigma_1
        if GP_collect_sigma_2 is None:
            self.GP_collect_sigma_2 = 0.
        else:
            self.GP_collect_sigma_2 = GP_collect_sigma_2

    def online_rth_aux_fun_params(self, x0):
        a_21 = self.robust_psi1_r_fun(x0[0], x0[1], x0[2], x0[3], x0[4], 0.0)
        a_11 = self.robust_psi1_r_fun(x0[0], x0[1], x0[2], x0[3], x0[4], 1.0) - a_21

        a_22 = self.robust_psi2_r_fun(x0[0], x0[1], x0[2], x0[3], x0[4], 0.0)
        a_12 = self.robust_psi2_r_fun(x0[0], x0[1], x0[2], x0[3], x0[4], 1.0) - a_22

        return a_11, a_21, a_12, a_22

    def online_true_rth_aux_fun_params(self, x0):
        a_21 = self.robust_true_psi1_r_fun(x0[0], x0[1], x0[2], x0[3], x0[4], 0.0)
        a_11 = self.robust_true_psi1_r_fun(x0[0], x0[1], x0[2], x0[3], x0[4], 1.0) - a_21

        a_22 = self.robust_true_psi2_r_fun(x0[0], x0[1], x0[2], x0[3], x0[4], 0.0)
        a_12 = self.robust_true_psi2_r_fun(x0[0], x0[1], x0[2], x0[3], x0[4], 1.0) - a_22

        return a_11, a_21, a_12, a_22

    def gp_collect_data(self, state_action, current_step, eating):
        self.X_ht.add(state_action)
        if current_step >= self.r:
            past_eating = eating[current_step-self.r]
            if not past_eating:
                previous_state_action = self.X_ht.history[-1]  # old action, 3 steps before the current one
                psi1_error = self.psi1_r.copy()
                psi2_error = self.psi2_r.copy()
                psi1_estimate = (self.no_meal_psi1_r_fun(*previous_state_action))
                psi2_estimate = (self.no_meal_psi2_r_fun(*previous_state_action))
                for i in range(self.state_dim):
                    for k in range(self.r+1):
                        psi1_error = psi1_error.subs({self.sym_state_to_subs[k+i*(self.r+1)]: self.X_ht.history[-(k+1), i]})
                        psi2_error = psi2_error.subs({self.sym_state_to_subs[k+i*(self.r+1)]: self.X_ht.history[-(k+1), i]})
                psi1_error -= psi1_estimate + np.random.normal(scale=self.GP_collect_sigma_1)
                psi2_error -= psi2_estimate + np.random.normal(scale=self.GP_collect_sigma_2)
                self.GP_memory.store(previous_state_action, psi1_error, psi2_error)


class OptimizationProblemWithHODTCBFAndGPRegression:
    def __init__(self, k_delta, K_1, K_2, max_action, min_action, gamma_1, gamma_2):
        self.k_delta = k_delta
        self.min_action = min_action
        self.max_action = max_action

        r = 1

        m = 4
        n = 4
        f = cp.Constant(np.array([0., 0., 0., 1.]))

        A_1 = cp.Constant(np.diag([1., np.sqrt(K_1), np.sqrt(K_2), 0.]))
        A_2 = cp.Constant(np.hstack((np.array([[2.]]), np.zeros((1, 3)))))
        A_31 = cp.Parameter(shape=(r + 1, 1), name='A_31')
        A_32 = cp.Constant(np.zeros((r + 1, 3)))
        A_3 = cp.hstack([A_31, A_32])
        A_41 = cp.Parameter(shape=(r + 1, 1), name='A_41')
        A_42 = cp.Constant(np.zeros((r + 1, 3)))
        A_4 = cp.hstack([A_41, A_42])

        A = [A_1, A_2, A_3, A_4]

        b_1 = cp.Constant(np.zeros(1))
        b_2 = cp.Parameter(shape=(1,), name='b_2')
        b_3 = cp.Parameter(shape=(r + 1,), name='b_3')
        b_4 = cp.Parameter(shape=(r + 1,), name='b_4')

        b = [b_1, b_2, b_3, b_4]

        c_1 = cp.Constant(np.concatenate((np.zeros(3), np.array([1.]))))
        c_2 = cp.Constant(np.zeros(n))
        c_31 = cp.Parameter(name="c_31")
        c_32 = cp.Constant(np.concatenate((np.array([math.prod(gamma_1)]), np.zeros(2))))
        c_3 = cp.hstack([c_31, c_32])
        c_41 = cp.Parameter(name="c_41")
        c_42 = cp.Constant(np.concatenate((np.zeros(1), np.array([math.prod(gamma_2)]), np.zeros(1))))
        c_4 = cp.hstack([c_41, c_42])

        c = [c_1, c_2, c_3, c_4]

        d_1 = cp.Constant(np.zeros(1))
        d_2 = cp.Constant(np.array([(self.max_action - self.min_action)]))
        d_3 = cp.Parameter(shape=(1,), name='d_3')
        d_4 = cp.Parameter(shape=(1,), name='d_4')

        d = [d_1, d_2, d_3, d_4]

        # Define and solve the CVXPY problem.
        x = cp.Variable(n, name='x')

        # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
        soc_constraints = [
            cp.SOC(c[j].T @ x + d[j], A[j] @ x + b[j]) for j in range(m)
        ]

        l_constraints = [
            x[-3] >= 0.,
            x[-2] >= 0.
        ]

        self.prob = cp.Problem(cp.Minimize(f.T @ x),
                               soc_constraints + l_constraints)
        # print(self.prob)
        # print("Is DPP? ", self.prob.is_dcp(dpp=True))
        # print("Is DCP? ", self.prob.is_dcp(dpp=False))

        self.A_32 = A_32
        self.A_42 = A_42
        self.c_32 = c_32
        self.c_42 = c_42

    def solve(self, a_11, a_21, a_12, a_22, psi1_mu_r, psi1_mu_m, psi2_mu_r, psi2_mu_m, psi1_LTr, psi1_LTm,
              psi2_LTr, psi2_LTm, u_RL):

        A_31 = self.k_delta * psi1_LTm
        A_41 = self.k_delta * psi2_LTm
        b_2 = np.array([2 * u_RL - (self.max_action + self.min_action)])
        b_3 = np.squeeze(self.k_delta * (psi1_LTr @ np.array([[1.]]) + psi1_LTm * u_RL))
        b_4 = np.squeeze(self.k_delta * (psi2_LTr @ np.array([[1.]]) + psi2_LTm * u_RL))
        c_31 = a_11 + np.squeeze(psi1_mu_m)
        c_41 = a_12 + np.squeeze(psi2_mu_m)
        d_3 = np.array([a_21 + np.squeeze(psi1_mu_r.T @ np.array([[1.]])) + (a_11 + np.squeeze(psi1_mu_m)) * u_RL])
        d_4 = np.array([a_22 + np.squeeze(psi2_mu_r.T @ np.array([[1.]])) + (a_12 + np.squeeze(psi2_mu_m)) * u_RL])

        self.prob.param_dict['A_31'].value = A_31
        self.prob.param_dict['A_41'].value = A_41
        self.prob.param_dict['b_2'].value = b_2
        self.prob.param_dict['b_3'].value = b_3
        self.prob.param_dict['b_4'].value = b_4
        self.prob.param_dict['c_31'].value = c_31
        self.prob.param_dict['c_41'].value = c_41
        self.prob.param_dict['d_3'].value = d_3
        self.prob.param_dict['d_4'].value = d_4

        self.prob.solve(solver='CLARABEL')

        x_opt = self.prob.var_dict['x'].value

        A_3 = np.hstack([A_31, self.A_32.value])
        A_4 = np.hstack([A_41, self.A_42.value])
        c_3 = np.hstack([c_31, self.c_32.value])
        c_4 = np.hstack([c_41, self.c_42.value])

        A_CBF = [A_3, A_4]
        b_CBF = [b_3, b_4]
        c_CBF = [c_3, c_4]
        d_CBF = [d_3, d_4]

        return x_opt, A_CBF, b_CBF, c_CBF, d_CBF
