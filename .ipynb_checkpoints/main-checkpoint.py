import numpy as np
from agent import Agent
from gps import (gp_delta_kernel, compute_mean_and_square_root_covariance,
                 preliminary_computations_for_mnsrc)
from utils import (plot_learning_curve, manage_memory, DT_Bergman_dynamics, plot_evaluation_run_with_GPs,
                   reward_calc, plot_reward_function, plot_violation_curve, meal_schedule, ExplorationPolicy,
                   plot_inside_safe_set_violations, plot_G_curve)
from cbfcontroller import (BergmanT1DDTHOCBFsUtils,
                           OptimizationProblemWithHODTCBFAndGPRegression)
import gpflow
import tensorflow as tf
import time
import os

if __name__ == '__main__':
    # Our knowledge about dynamics: nominal model dynamics
    nominal_model_params = {
        'p_1': 2.3e-6,
        'G_b': 75.0,
        'p_2': 0.088,
        'p_3': 0.63e-3,
        'n': 0.09,
        'I_b': 15.0,
        'tau_G': 47.0,
        'V_G': 253.0
    }

    # gamma_1 smaller values than gamma_2 because not only I need to avoid hypoglycemia at all costs,
    # but, most importantly, because action_max is well enough to deal with any BG spike, while action_min cannot deal
    # with any trough the system will face, it would hope to provide negative values, but the constraint force it not.
    gamma_1 = [0.20, 0.15, 0.10]

    gamma_2 = [0.35, 0.30, 0.25]

    np.random.seed(43)

    # the true model dynamics
    true_model_params = {
        k: v if k in ('tau_G', 'V_G')
        else np.random.choice([0.7, 1.3]) * v
        for k, v in nominal_model_params.items()
    }
    print(f'\nThe true model parameters:\n{true_model_params}\n')

    # every time should be thought in minutes, unless specified differently
    dt = 1  # 0 < dt < 60

    N = 24 * 60
    print(f'Total number of steps: {N}.')
    CHO_max = 65_000  # YOU ASSUME TO KNOW THE MAXIMAL CHO INTAKE

    np.random.seed(None)

    answer1 = input("Do you want to do an evaluation run? (y/n) ")
    while not (answer1 == 'yes' or answer1 == 'Yes' or answer1 == 'YES' or answer1 == 'y' or answer1 == 'Y'
               or answer1 == 'no' or answer1 == 'No' or answer1 == 'NO' or answer1 == 'n' or answer1 == 'N'):
        answer1 = input("Please provide a yes or a no as an answer! ")
    if answer1 == 'yes' or answer1 == 'Yes' or answer1 == 'YES' or answer1 == 'y' or answer1 == 'Y':
        evaluate = True
        restore_training = False
    else:
        evaluate = False
        answer2 = input("Do you want to do restore training? (y/n) ")
        while not (answer2 == 'yes' or answer2 == 'Yes' or answer2 == 'YES' or answer2 == 'y' or answer2 == 'Y'
                   or answer2 == 'no' or answer2 == 'No' or answer2 == 'NO' or answer2 == 'n' or answer2 == 'N'):
            answer2 = input("Please provide a yes or a no as an answer! ")
        if answer2 == 'yes' or answer2 == 'Yes' or answer2 == 'YES' or answer2 == 'y' or answer2 == 'Y':
            restore_training = True
        else:
            restore_training = False

    action_dim = 1
    state_dim = 5
    max_action = 30.  # important to guarantee it's a float
    min_action = 0.

    max_action_agent = 0.3
    min_action_agent = min_action
    size_action_agent = (max_action_agent - min_action_agent) / 2
    center_action_agent = (max_action_agent + min_action_agent) / 2

    manage_memory()
    agent = Agent(state_dim=state_dim,
                  action_dim=action_dim,
                  max_action=max_action_agent,
                  min_action=min_action_agent,
                  alr=1e-4, clr=1e-3,
                  max_size=1_000_000, tau=5e-3, d=2, explore_sigma=0.1 * size_action_agent,
                  smooth_sigma=0.2 * size_action_agent, c=0.5 * size_action_agent, fc1_dims=400, fc2_dims=300,
                  batch_size=128)
    os.makedirs(agent.chkpt_dir, exist_ok=True)

    GP_max_size = 1000
    state_normalizer = tf.keras.layers.Normalization()

    hodtcbfs_utils = BergmanT1DDTHOCBFsUtils(gamma_1, gamma_2, dt, CHO_max, nominal_model_params, true_model_params,
                                             GP_max_size, N, GP_collect_sigma_1=0.01, GP_collect_sigma_2=0.1)

    kernels_psi = gp_delta_kernel(state_dim, action_dim)
    kernels_phi = gp_delta_kernel(state_dim, action_dim)
    GP_psi_dir = 'models/GP_psi'
    GP_phi_dir = 'models/GP_phi'

    k_delta = 1.5
    K_1 = 1e6
    K_2 = 1e6
    optproblem = OptimizationProblemWithHODTCBFAndGPRegression(k_delta, K_1, K_2, max_action, min_action,
                                                               gamma_1, gamma_2)
    G0 = 140
    best_score = reward_calc(50) * N  # value that will get updated ofc
    print(f'Worst case scenario score: {best_score}.')
    print(f'Best policy reward: {reward_calc(G0) * N}.')  # best reward we can hope
    reward_avg_window = 50
    score_history = []
    max_violation_history = []
    min_G_history = []
    max_G_history = []

    max_games_to_plot = 5

    os.makedirs('data', exist_ok=True)

    if evaluate:
        n_games = 1

        agent.load_models()

    else:
        n_games = 500
        print(f'Number of episodes: {n_games}.')
        os.makedirs("plots", exist_ok=True)
        plot_reward_function(figure_file='plots/RewardFunction.pdf')
        if restore_training:
            agent.load_models()

        final_sigma = 1e-4
        exploration_policy = ExplorationPolicy(0.1 * size_action_agent, n_games, final_sigma, 1e-6, 20)

    time.sleep(5)

    for j in range(n_games):
        g_state = np.clip(np.random.normal(loc=G0, scale=2.5), a_min=G0 - 5, a_max=G0 + 5)
        print(f'Episode {j} started.')
        CHOs, eating = meal_schedule(N, dt, CHO_max)

        if evaluate or (j == 0 and restore_training):
            psi_model = tf.saved_model.load(GP_psi_dir + '/restored')
            phi_model = tf.saved_model.load(GP_phi_dir + '/restored')
        elif j > 0:
            psi_model = tf.saved_model.load(GP_psi_dir)
            phi_model = tf.saved_model.load(GP_phi_dir)

        score = 0
        episode_violations = []
        episode_Gs = []
        state = np.array([g_state, 0., true_model_params['I_b'], 0., 0.])
        if evaluate:
            explore_sigma = None
        else:
            explore_sigma = exploration_policy.get_sigma()

        states = []
        controls = []
        worst_CBF_psi = []
        worst_CBF_phi = []
        mean_CBF_psi = []
        mean_CBF_phi = []
        nominal_CBF_psi = []
        nominal_CBF_phi = []
        epsilon_psi = []
        epsilon_phi = []
        true_CBF_psi = []
        true_CBF_phi = []
        check_CBF_psi = []
        check_CBF_phi = []

        epsilon_max_psi = 0.
        epsilon_max_phi = 0.
        hodtcbfs_utils.reset_collect()
        
        for i in range(N):
            if j < max_games_to_plot:
                hodtcbfs_utils.collect(state)
                if hodtcbfs_utils.collect_cntr == hodtcbfs_utils.r:
                    if not (hodtcbfs_utils.safe_set1_check() and hodtcbfs_utils.safe_set2_check()):
                        raise SystemExit(f'The system does not start in the safe set in episode {i}!')

            episode_violations.append(max(0., - state[0] + hodtcbfs_utils.BG_min, + state[0] - hodtcbfs_utils.BG_max))
            episode_Gs.append(state[0])

            u_RL = agent.choose_action(state, evaluate, explore_sigma)[0].numpy()

            if j > 0 or evaluate or restore_training:

                (psi_m_r_temp, psi_m_1_temp, psi_Lr_bar_temp,
                 psi_L1_bar_temp) = psi_model.compiled_mean_and_square_root_covariance(state.reshape((1, 5)))

                if tf.reduce_any(tf.stack([tf.reduce_any(tf.math.is_nan(psi_Lr_bar_temp)),
                                           tf.reduce_any(tf.math.is_nan(psi_L1_bar_temp))])):
                    tf.print("\n!!! Cholesky produced NaNs for Sigma_psi!")

                (phi_m_r_temp, phi_m_1_temp, phi_Lr_bar_temp,
                 phi_L1_bar_temp) = phi_model.compiled_mean_and_square_root_covariance(state.reshape((1, 5)))

                if tf.reduce_any(tf.stack([tf.reduce_any(tf.math.is_nan(phi_Lr_bar_temp)),
                                           tf.reduce_any(tf.math.is_nan(phi_L1_bar_temp))])):
                    tf.print("\n!!! Cholesky produced NaNs for Sigma_phi!")

                psi_m_r, psi_m_1, psi_Lr_bar, psi_L1_bar = (psi_m_r_temp.numpy(), psi_m_1_temp.numpy(),
                                                            psi_Lr_bar_temp.numpy(), psi_L1_bar_temp.numpy())

                phi_m_r, phi_m_1, phi_Lr_bar, phi_L1_bar = (phi_m_r_temp.numpy(), phi_m_1_temp.numpy(),
                                                            phi_Lr_bar_temp.numpy(), phi_L1_bar_temp.numpy())

            else:  # basically if we are in the first step of the training phase and not in evaluation,
                # we restrict ourselves to a simple QP
                psi_m_r, psi_m_1, psi_Lr_bar, psi_L1_bar = (np.zeros((1, 1)),
                                                            np.zeros((1, action_dim)), np.zeros((1 + action_dim, 1)),
                                                            np.zeros((1 + action_dim, action_dim)))
                phi_m_r, phi_m_1, phi_Lr_bar, phi_L1_bar = (np.zeros((1, 1)),
                                                            np.zeros((1, action_dim)), np.zeros((1 + action_dim, 1)),
                                                            np.zeros((1 + action_dim, action_dim)))

            a_11, a_21, a_12, a_22 = hodtcbfs_utils.online_rth_aux_fun_params(state)

            sol, A_CBF, b_CBF, c_CBF, d_CBF = optproblem.solve(a_11, a_21, a_12, a_22, psi_m_r, psi_m_1, phi_m_r,
                                                               phi_m_1, psi_Lr_bar, psi_L1_bar, phi_Lr_bar, phi_L1_bar,
                                                               u_RL)

            u_CBF = sol[0]

            print(f'Step {i} (ep. {j}) BG level: {state[0]:.5f} | u_CBF: {u_CBF:.5f} | u_RL: {u_RL:.5f}')

            if evaluate:
                with open('evaluation_run_details.txt', 'w') as f:
                    f.write(f'Step {i} (ep. {j}) BG level: {state[0]:.5f} | u_CBF: {u_CBF:.5f} | u_RL: {u_RL:.5f}\n')

            ID = np.clip(u_RL + u_CBF, a_min=min_action, a_max=max_action)  # pump limit
            control = np.array([ID, CHOs[i, 0]])

            if j < max_games_to_plot:
                sol_no_violations = sol.copy()
                sol_no_violations[-3] = 0.
                sol_no_violations[-2] = 0.

                states.append(state)
                controls.append(control)
                worst_CBF_psi.append(np.squeeze(c_CBF[0].T @ sol_no_violations + d_CBF[0] -
                                                np.linalg.norm(A_CBF[0] @ sol_no_violations + b_CBF[0])))
                worst_CBF_phi.append(np.squeeze(c_CBF[1].T @ sol_no_violations + d_CBF[1] -
                                                np.linalg.norm(A_CBF[1] @ sol_no_violations + b_CBF[1])))
                mean_CBF_psi.append(np.squeeze(c_CBF[0].T @ sol_no_violations + d_CBF[0]))
                mean_CBF_phi.append(np.squeeze(c_CBF[1].T @ sol_no_violations + d_CBF[1]))
                nominal_CBF_psi.append(a_11 * (sol_no_violations[0] + u_RL) + a_21)
                nominal_CBF_phi.append(a_12 * (sol_no_violations[0] + u_RL) + a_22)
                epsilon_psi.append(sol[-3])
                epsilon_phi.append(sol[-2])
                at_11, at_21, at_12, at_22 = hodtcbfs_utils.online_true_rth_aux_fun_params(state)
                true_CBF_psi.append(at_11 * (sol_no_violations[0] + u_RL) + at_21)
                true_CBF_phi.append(at_12 * (sol_no_violations[0] + u_RL) + at_22)

                if hodtcbfs_utils.collect_cntr >= hodtcbfs_utils.r:

                    if hodtcbfs_utils.safe_set1_check():
                        if epsilon_psi[- hodtcbfs_utils.r] >= epsilon_max_psi:
                            epsilon_max_psi = epsilon_psi[- hodtcbfs_utils.r]
                        check_CBF_psi.append(1.)
                    else:
                        if hodtcbfs_utils.safe_set1_check(epsilon_max_psi):
                            check_CBF_psi.append(0.5)
                        else:
                            check_CBF_psi.append(0.)

                    if hodtcbfs_utils.safe_set2_check():
                        if epsilon_phi[- hodtcbfs_utils.r] >= epsilon_max_phi:
                            epsilon_max_phi = epsilon_phi[- hodtcbfs_utils.r]
                        check_CBF_phi.append(1.)
                    else:
                        if hodtcbfs_utils.safe_set2_check(epsilon_max_phi):
                            check_CBF_phi.append(0.5)
                        else:
                            check_CBF_phi.append(0.)

            new_state = DT_Bergman_dynamics(state, control, true_model_params, dt)
            if new_state[0] < 50 or new_state[0] > 250:
                print('Try again! Blood glucose level went to an overly dangerous hyper/hypoglycemia level!')
                raise SystemExit

            reward = reward_calc(new_state[0])

            if not evaluate:
                state_action = np.concatenate((state, control[:1]), axis=0)  # action needs to be (1,)

                hodtcbfs_utils.gp_collect_data(state_action, i, eating)

                if restore_training or j > 0:
                    agent.collect_transition(state, u_RL, reward, new_state, done=False)
                    # done=True --> for transitions where the episode terminates by reaching some failure state,
                    # and not due to the episode running until the max horizon (TD3 paper appendix)
                    agent.learn()

            score += reward
            state = new_state
            # END EPISODE CYCLE

        if j < max_games_to_plot:
            np.save(f'data/states-Evaluate{evaluate}-Episode{j}.npy', np.array(states))
            np.save(f'data/controls-Evaluate{evaluate}-Episode{j}.npy', np.array(controls))
            np.save(f'data/worst_CBF_psi-Evaluate{evaluate}-Episode{j}.npy', np.array(worst_CBF_psi))
            np.save(f'data/worst_CBF_phi-Evaluate{evaluate}-Episode{j}.npy', np.array(worst_CBF_phi))
            np.save(f'data/mean_CBF_psi-Evaluate{evaluate}-Episode{j}.npy', np.array(mean_CBF_psi))
            np.save(f'data/mean_CBF_phi-Evaluate{evaluate}-Episode{j}.npy', np.array(mean_CBF_phi))
            np.save(f'data/nominal_CBF_psi-Evaluate{evaluate}-Episode{j}.npy', np.array(nominal_CBF_psi))
            np.save(f'data/nominal_CBF_phi-Evaluate{evaluate}-Episode{j}.npy', np.array(nominal_CBF_phi))
            np.save(f'data/epsilon_psi-Evaluate{evaluate}-Episode{j}.npy', np.array(epsilon_psi))
            np.save(f'data/epsilon_phi-Evaluate{evaluate}-Episode{j}.npy', np.array(epsilon_phi))
            np.save(f'data/true_CBF_psi-Evaluate{evaluate}-Episode{j}.npy', np.array(true_CBF_psi))
            np.save(f'data/true_CBF_phi-Evaluate{evaluate}-Episode{j}.npy', np.array(true_CBF_phi))
            np.save(f'data/check_CBF_psi-Evaluate{evaluate}-Episode{j}.npy', np.array(check_CBF_psi))
            np.save(f'data/check_CBF_phi-Evaluate{evaluate}-Episode{j}.npy', np.array(check_CBF_phi))

            plot_evaluation_run_with_GPs(dt, states, controls, worst_CBF_psi, worst_CBF_phi,
                                         mean_CBF_psi, mean_CBF_phi, nominal_CBF_psi, nominal_CBF_phi,
                                         epsilon_psi, epsilon_phi, k_delta, true_CBF_psi, true_CBF_phi,
                                         figure_file=f'plots/BergmanRunTD3-Evaluate{evaluate}-Episode{j}'
                                                     f'-Score{score:.0f}.pdf')
            plot_inside_safe_set_violations(dt, epsilon_psi, check_CBF_psi,
                                            figure_file=f'plots/SafeSetViolationsPsi-Evaluate{evaluate}-Episode{j}.pdf')
            plot_inside_safe_set_violations(dt, epsilon_phi, check_CBF_phi,
                                            figure_file=f'plots/SafeSetViolationsPhi-Evaluate{evaluate}-Episode{j}.pdf')

        if evaluate:
            print(f'Episode terminated. Score {score:.1f}. Max violation {max(episode_violations):.1f}.\n')
        else:
            max_violation_history.append(max(episode_violations))
            score_history.append(score)  # do not consider the score at step 0 in avg score! Modify it.
            max_G_history.append(max(episode_Gs))
            min_G_history.append(min(episode_Gs))
            avg_score = np.mean(score_history[-reward_avg_window:])
            print(f'Episode {j} terminated. Score {score:.1f}. Avg score {avg_score:.1f}.'
                  f' Max violation {max(episode_violations):.1f}.\n')

            agent_saved = False
            if avg_score > best_score:
                best_score = avg_score
                agent_saved = agent.save_models()
                print(f"Is the agent model saved? {agent_saved}.\n")

            X, Y_psi, Y_phi = hodtcbfs_utils.GP_memory.special_sample_buffer()

            _, idx = np.unique(X, axis=0, return_index=True)
            if len(idx) < X.shape[0]:
                print("Beware: in X there are", X.shape[0] - len(idx), "duplicated rows.")
            # print(f'X shape: {X.shape}, take a look: {X[:10, :]}\n')
            # print(f'Y_psi shape: {Y_psi.shape}, take a look: {Y_psi[:10, :]}\n')
            # print(f'Y_phi shape: {Y_phi.shape}, take a look: {Y_phi[:10, :]}\n')

            state_normalizer.adapt(X[:, :state_dim])

            X_ext = np.concatenate((state_normalizer(X[:, :state_dim]), np.ones((X.shape[0], 1)),
                                    X[:, state_dim:state_dim + action_dim]), axis=1)

            psi_model = gpflow.models.GPR((X_ext, Y_psi), kernel=kernels_psi)
            phi_model = gpflow.models.GPR((X_ext, Y_phi), kernel=kernels_phi)

            opt = gpflow.optimizers.Scipy()
            opt.minimize(psi_model.training_loss, psi_model.trainable_variables)
            opt.minimize(phi_model.training_loss, phi_model.trainable_variables)

            gpflow.utilities.print_summary(psi_model)
            gpflow.utilities.print_summary(phi_model)

            beta_rows, psi_m_right_factor, psi_L_hat = preliminary_computations_for_mnsrc(psi_model)
            _, phi_m_right_factor, phi_L_hat = preliminary_computations_for_mnsrc(phi_model)

            psi_model.state_normalizer = state_normalizer
            phi_model.state_normalizer = state_normalizer

            psi_model.compiled_mean_and_square_root_covariance = tf.function(
                lambda x: compute_mean_and_square_root_covariance(x, model=psi_model, beta_rows=beta_rows,
                                                                  m_right_factor=psi_m_right_factor, L_hat=psi_L_hat,
                                                                  action_dim=action_dim),
                input_signature=[tf.TensorSpec(shape=[1, state_dim], dtype=tf.float64)],
            )

            phi_model.compiled_mean_and_square_root_covariance = tf.function(
                lambda x: compute_mean_and_square_root_covariance(x, model=phi_model, beta_rows=beta_rows,
                                                                  m_right_factor=phi_m_right_factor, L_hat=phi_L_hat,
                                                                  action_dim=action_dim),
                input_signature=[tf.TensorSpec(shape=[1, state_dim], dtype=tf.float64)],
            )

            tf.saved_model.save(psi_model, GP_psi_dir)
            tf.saved_model.save(phi_model, GP_phi_dir)
            if agent_saved:
                tf.saved_model.save(psi_model, GP_psi_dir + '/restored')
                tf.saved_model.save(phi_model, GP_phi_dir + '/restored')

        # END GAMES CYCLE

    if not evaluate:
        np.save(r'data/max_G_history.npy', np.array(max_G_history))
        np.save(r'data/min_G_history.npy', np.array(min_G_history))
        np.save(r'data/score_history.npy', np.array(score_history))
        np.save(r'data/max_violation_history.npy', np.array(max_violation_history))
        x = [i + 1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, reward_avg_window, figure_file='plots/BergmanTrainingScoreTD3.pdf')
        plot_violation_curve(x, max_violation_history, figure_file='plots/BergmanTrainingViolationsTD3.pdf')
        plot_G_curve(x, max_G_history, min_G_history, figure_file='plots/BergmanTrainingGTrend.pdf')