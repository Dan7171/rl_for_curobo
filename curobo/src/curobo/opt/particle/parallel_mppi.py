#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Standard Library
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.opt.particle.particle_opt_base import ParticleOptBase, ParticleOptConfig, SampleMode
from curobo.opt.particle.particle_opt_utils import (
    SquashType,
    cost_to_go,
    gaussian_entropy,
    matrix_cholesky,
    scale_ctrl,
)
from curobo.rollout.rollout_base import RolloutBase, Trajectory
from curobo.types.base import TensorDeviceType
from curobo.types.robot import State
from curobo.util.logger import log_info
from curobo.util.sample_lib import HaltonSampleLib, SampleConfig, SampleLib
from curobo.util.tensor_util import copy_tensor
from curobo.util.torch_utils import get_torch_jit_decorator


class BaseActionType(Enum):
    REPEAT = 0
    NULL = 1
    RANDOM = 2


class CovType(Enum):
    SIGMA_I = 0
    DIAG_A = 1
    FULL_A = 2
    FULL_HA = 3


@dataclass
class ParallelMPPIConfig(ParticleOptConfig):
    init_mean: float
    init_cov: float
    base_action: BaseActionType
    step_size_mean: float
    step_size_cov: float
    null_act_frac: float
    squash_fn: SquashType
    cov_type: CovType
    sample_params: SampleConfig
    update_cov: bool
    random_mean: bool
    beta: float
    alpha: float
    gamma: float
    kappa: float
    sample_per_problem: bool

    def __post_init__(self):
        self.init_cov = self.tensor_args.to_device(self.init_cov).unsqueeze(0)
        self.init_mean = self.tensor_args.to_device(self.init_mean).clone()
        return super().__post_init__()

    @staticmethod
    @profiler.record_function("parallel_mppi_config/create_data_dict")
    def create_data_dict(
        data_dict: Dict,
        rollout_fn: RolloutBase,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        child_dict: Optional[Dict] = None,
    ):
        if child_dict is None:
            child_dict = deepcopy(data_dict)
        child_dict = ParticleOptConfig.create_data_dict(
            data_dict, rollout_fn, tensor_args, child_dict
        )
        child_dict["base_action"] = BaseActionType[child_dict["base_action"]]
        child_dict["squash_fn"] = SquashType[child_dict["squash_fn"]]
        child_dict["cov_type"] = CovType[child_dict["cov_type"]]
        child_dict["sample_params"]["d_action"] = rollout_fn.d_action
        child_dict["sample_params"]["horizon"] = rollout_fn.action_horizon
        child_dict["sample_params"]["tensor_args"] = tensor_args
        child_dict["sample_params"] = SampleConfig(**child_dict["sample_params"])

        # init_mean:
        if "init_mean" not in child_dict:
            child_dict["init_mean"] = rollout_fn.get_init_action_seq()
        return child_dict


class ParallelMPPI(ParticleOptBase, ParallelMPPIConfig):
    @profiler.record_function("parallel_mppi/init")
    def __init__(self, config: Optional[ParallelMPPIConfig] = None):
        if config is not None:
            ParallelMPPIConfig.__init__(self, **vars(config))
        ParticleOptBase.__init__(self)

        self.sample_lib = SampleLib(self.sample_params)
        self._sample_set = None
        self._sample_iter = None
        # initialize covariance types:
        if self.cov_type == CovType.FULL_HA:
            self.I = torch.eye(
                self.action_horizon * self.d_action,
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )

        else:  # AxA
            self.I = torch.eye(
                self.d_action, device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )

        self.Z_seq = torch.zeros(
            1,
            self.action_horizon,
            self.d_action,
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )

        self.delta = None
        self.mean_action = None
        self.act_seq = None
        self.cov_action = None
        self.best_traj = None
        self.scale_tril = None
        self.visual_traj = None
        if self.debug_info is not None and "visual_traj" in self.debug_info.keys():
            self.visual_traj = self.debug_info["visual_traj"]
            self.k_rollouts_debug = 20 # original value
            if "k" in self.debug_info.keys():
                self.k_rollouts_debug = self.debug_info["k"] # number of top trajectories to store
                if self.k_rollouts_debug == 'all' or self.k_rollouts_debug > self.total_num_particles:
                    self.k_rollouts_debug = self.total_num_particles
        self.top_values = None
        self.top_idx = None
        self.top_trajs = None

        self.mean_lib = HaltonSampleLib(
            SampleConfig(
                self.action_horizon,
                self.d_action,
                tensor_args=self.tensor_args,
                **{"fixed_samples": False, "seed": 2567, "filter_coeffs": None}
            )
        )

        self.reset_distribution()
        self.update_samples()
        self._use_cuda_graph = False
        self._init_cuda_graph = False
        self.info = dict(rollout_time=0.0, entropy=[])
        self._batch_size = -1
        self._store_debug = False

    def get_rollouts(self):
        return self.top_trajs

    def reset_distribution(self):
        """
        Reset control distribution
        """

        self.reset_mean()
        self.reset_covariance()

    def _compute_total_cost(self, costs):
        """
        Calculate weights using exponential utility
        """

        # cost_seq = self.gamma_seq * costs
        # cost_seq = torch.sum(cost_seq, dim=-1, keepdim=False) / self.gamma_seq[..., 0]
        # print(self.gamma_seq.shape, costs.shape)
        cost_seq = jit_compute_total_cost(self.gamma_seq, costs)
        return cost_seq

    def _exp_util(self, total_costs):
        w = jit_calculate_exp_util(self.beta, total_costs)
        # w = torch.softmax((-1.0 / self.beta) * total_costs, dim=-1)
        return w

    def _exp_util_from_costs(self, costs):
        # col_mask = getattr(self.rollout_fn, 'current_collision_mask', None)
        w = jit_calculate_exp_util_from_costs(costs, self.gamma_seq, self.beta) #, col_mask)
        return w

    def _compute_mean(self, w, actions):
        # get the new means from here
        new_mean = torch.sum(w * actions, dim=-3)
        new_mean = jit_blend_mean(self.mean_action, new_mean, self.step_size_mean)
        return new_mean

    def _compute_mean_covariance(self, costs, actions):
        if self.cov_type == CovType.FULL_A:
            log_error("Not implemented")
        if self.cov_type == CovType.DIAG_A:
            new_mean, new_cov, new_scale_tril = jit_mean_cov_diag_a(
                costs,
                actions,
                self.gamma_seq,
                self.mean_action,
                self.cov_action,
                self.step_size_mean,
                self.step_size_cov,
                self.kappa,
                self.beta,
            )
            self.scale_tril.copy_(new_scale_tril)
            # self._update_cov_scale(new_cov)

        else:
            w = self._exp_util_from_costs(costs)
            w = w.unsqueeze(-1).unsqueeze(-1)
            new_mean = self._compute_mean(w, actions)
            new_cov = self._compute_covariance(w, actions)
            self._update_cov_scale(new_cov)

        return new_mean, new_cov

    def _compute_covariance(self, w, actions):
        if not self.update_cov:
            return
        # w = w.squeeze(-1).squeeze(-1)
        # w = w[0, :]
        if self.cov_type == CovType.SIGMA_I:
            delta_actions = actions - self.mean_action.unsqueeze(-3)

            weighted_delta = w * (delta_actions**2)
            cov_update = torch.mean(
                torch.sum(torch.sum(weighted_delta, dim=-2), dim=-1), dim=-1, keepdim=True
            )

        elif self.cov_type == CovType.DIAG_A:

            cov_update = jit_diag_a_cov_update(w, actions, self.mean_action)

        elif self.cov_type == CovType.FULL_A:
            delta_actions = actions - self.mean_action.unsqueeze(-3)

            delta = delta_actions[0, ...]

            raise NotImplementedError
        elif self.cov_type == CovType.FULL_HA:
            delta_actions = actions - self.mean_action.unsqueeze(-3)

            delta = delta_actions[0, ...]

            weighted_delta = (
                torch.sqrt(w) * delta.view(delta.shape[0], delta.shape[1] * delta.shape[2]).T
            )  # .unsqueeze(-1)
            cov_update = torch.matmul(weighted_delta, weighted_delta.T)

        else:
            raise ValueError("Unidentified covariance type in update_distribution")
        cov_update = jit_blend_cov(self.cov_action, cov_update, self.step_size_cov, self.kappa)
        return cov_update

    def _update_cov_scale(self, new_cov=None):
        if new_cov is None:
            new_cov = self.cov_action
        if not self.update_cov:
            return
        if self.cov_type == CovType.SIGMA_I:
            self.scale_tril = torch.sqrt(new_cov)

        elif self.cov_type == CovType.DIAG_A:
            self.scale_tril.copy_(torch.sqrt(new_cov))

        elif self.cov_type == CovType.FULL_A:
            self.scale_tril = matrix_cholesky(new_cov)

        elif self.cov_type == CovType.FULL_HA:
            raise NotImplementedError

    @torch.no_grad()
    def _update_distribution(self, trajectories: Trajectory):
        costs = trajectories.costs
        actions = trajectories.actions

        # Let's reshape to n_problems now:

        # first find the means before doing exponential utility:

        with profiler.record_function("mppi/get_best"):

            # Update best action
            if self.sample_mode == SampleMode.BEST:
                w = self._exp_util_from_costs(costs)
                col_mask = getattr(self.rollout_fn, 'current_collision_mask', None)
                a_selected_from_cfree = False
                if col_mask is not None:
                        
                    #[B x Nrollouts] discounted cost sum over horizon and mean across particles (normally b = 1)
                  
                    cfree_rollout_mask = (torch.sum(col_mask, dim=-1) == 0).unsqueeze(0) # 1 where no collision in rollout, 0 elsewhere
                    cfree_w = w[cfree_rollout_mask]
                    if len(cfree_w) > 0:
                        best_idx = torch.argmax(cfree_w, dim=-1)
                        a_selected_from_cfree = True

                    # # print(f'DEBUG: len(cfree_rollouts): {len(cfree_rollouts)}, len(col_rollouts): {len(col_rollouts)}')
                    # if len(cfree_rollouts) > 0 and len(col_rollouts) > 0:
                    #     max_cfree_cost = torch.max(cfree_rollouts)
                    #     min_col_cost = torch.min(col_rollouts)
                    #     # print(f"DEBUG: max_cfree_cost: {max_cfree_cost}, min_col_cost: {min_col_cost}")
                    #     k = torch.max(torch.tensor(0.0), max_cfree_cost - min_col_cost) + 1000 
                    #     # print(f"DEBUG adding k: {k} to col rollouts: (N_col_rollouts: {len(col_rollouts)}), N_cfree_rollouts: {len(cfree_rollouts)}")
                    #     cost_seq[col_rollout_mask] = cost_seq[col_rollout_mask] + k # add k to col rollouts to make them larger than col free rollouts
                        
                if not a_selected_from_cfree:
                    best_idx = torch.argmax(w, dim=-1)
                self.best_traj.copy_(actions[self.problem_col, best_idx])
                print(f"DEBUG: best idx: {best_idx}, min w: {w.min()}, max w: {w.max()}, sum w: {w.sum()}")
        
        with profiler.record_function("mppi/store_rollouts"):

            if self.store_rollouts and self.visual_traj is not None:
                total_costs = self._compute_total_cost(costs)
                vis_seq = getattr(trajectories.state, self.visual_traj)
                # top_values, top_idx = torch.topk(total_costs, 20, dim=1)
                top_values, top_idx = torch.topk(total_costs, self.k_rollouts_debug, dim=1)
                self.top_values = top_values
                self.top_idx = top_idx
                top_trajs = torch.index_select(vis_seq, 0, top_idx[0])
                for i in range(1, top_idx.shape[0]):
                    trajs = torch.index_select(
                        vis_seq, 0, top_idx[i] + (self.particles_per_problem * i)
                    )
                    top_trajs = torch.cat((top_trajs, trajs), dim=0)
                if self.top_trajs is None or top_trajs.shape != self.top_trajs:
                    self.top_trajs = top_trajs
                else:
                    self.top_trajs.copy_(top_trajs)

        if not self.update_cov:
            w = self._exp_util_from_costs(costs)
            w = w.unsqueeze(-1).unsqueeze(-1)
            new_mean = self._compute_mean(w, actions)
        else:
            new_mean, new_cov = self._compute_mean_covariance(costs, actions)
            self.cov_action.copy_(new_cov)

        self.mean_action.copy_(new_mean)

    @torch.no_grad()
    def sample_actions(self, init_act):
        """
        This is a core function in MPPI that generates all the candidate action sequences (particles) for evaluation.
        """

        # Step 1: Get Pre-computed Random Samples
        # Gets noise samples from the pre-computed _sample_set
        # _sample_iter tracks which batch of samples to use (for reproducibility)
        # delta shape: [n_problems, particles_per_problem, action_horizon, d_action]
        # These are raw noise samples (e.g., from Halton sequences, random, etc.)
        delta = torch.index_select(self._sample_set, 0, self._sample_iter).squeeze(0)
        
        
        # Step 2: Update Sample Iterator
        # Advances the sample iterator to get different noise each iteration
        # When fixed_samples=True: uses same noise every time (for debugging/reproducibility)
        # When fixed_samples=False: cycles through different noise samples
        # Resets to 0 after n_iters to avoid running out of pre-computed samples
        if not self.sample_params.fixed_samples:
            self._sample_iter[:] += 1
            self._sample_iter_n += 1
            if self._sample_iter_n >= self.n_iters:
                self._sample_iter_n = 0
                self._sample_iter[:] = 0
                log_info(
                    "Resetting sample iterations in particle opt base to 0, this is okay during graph capture"
                )

        # Step 3: Scale the Noise
        # Scales the unit noise by the covariance structure
        # self.full_scale_tril is the Cholesky decomposition of the covariance matrix
        # This transforms standard noise into noise with the desired covariance
        # Result: Noise that respects the learned covariance structure
        scaled_delta = delta * self.full_scale_tril
        
        
        # Step 4: Add Noise to Mean
        # Creates candidate actions: mean + scaled_noise
        # mean_action.unsqueeze(-3) adds particle dimension: [n_problems, 1, action_horizon, d_action]
        # Broadcasting adds scaled noise to each mean
        # Result: Main particle samples centered around the mean


        act_seq = self.mean_action.unsqueeze(-3) + scaled_delta
        cat_list = [act_seq]

        # Step 5: Add Special Particles
        # Two types of special particles:
        # Negative Particles (neg_per_problem > 0):
        # Purpose: Exploration in opposite direction
        # Action: -1.0 * mean_action (negative of current mean)
        # Why useful: Helps escape local minima by trying opposite actions
        # Null Particles (null_per_problem > 0):
        # Purpose: Conservative baseline
        # Action: self.null_act_seqs (typically zeros - no action)
        # Why useful: Provides a "do nothing" option for safety
        
        if self.neg_per_problem > 0:
            neg_action = -1.0 * self.mean_action
            neg_act_seqs = neg_action.unsqueeze(-3).expand(-1, self.neg_per_problem, -1, -1)
            cat_list.append(neg_act_seqs)
        if self.null_per_problem > 0:
            cat_list.append(
                self.null_act_seqs[: self.null_per_problem]
                .unsqueeze(0)
                .expand(self.n_problems, -1, -1, -1)
            )

        # Step 6: Concatenate All Particles
        # Combines all particle types: [noisy_samples, negative_samples, null_samples]
        # Reshapes from structured format to flat batch format
        # Final shape: [total_particles, action_horizon, d_action]
        act_seq = torch.cat(
            (cat_list),
            dim=-3,
        )
        act_seq = act_seq.reshape(self.total_num_particles, self.action_horizon, self.d_action)
        
        # Step 7: Apply Action Bounds
        # Clips/squashes actions to respect joint limits
        # squash_fn determines how (clamp, tanh, etc.)
        # Ensures all sampled actions are physically feasible
        act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs, squash_fn=self.squash_fn)

        # if not copy_tensor(act_seq, self.act_seq):
        #    self.act_seq = act_seq
        return act_seq  # self.act_seq

    def update_seed(self, init_act):
        """
        What it does:
        Updates self.mean_action: Sets the center of the sampling distribution to init_act
        Updates self.best_traj: Also initializes the best trajectory found so far
        Handles batch dimensions: Expands init_act if needed to match n_problems
        Why this matters:
        Seeds the optimization with a good starting point (often the shifted previous solution)
        Centers the random sampling around this initial guess
        Provides warm-start - instead of sampling randomly, sample around the previous solution
        """
        self.update_init_mean(init_act)

    def update_init_mean(self, init_mean):
        # update mean:
        # init_mean = init_mean.clone()
        if init_mean.shape[0] != self.n_problems:
            init_mean = init_mean.expand(self.n_problems, -1, -1)
        if not copy_tensor(init_mean, self.mean_action):
            self.mean_action = init_mean.clone()
        if not copy_tensor(init_mean, self.best_traj):
            self.best_traj = init_mean.clone()

    def reset_mean(self):
        with profiler.record_function("mppi/reset_mean"):
            if self.random_mean:
                mean = self.mean_lib.get_samples([self.n_problems])
                self.update_init_mean(mean)
            else:
                self.update_init_mean(self.init_mean)

    def reset_covariance(self):
        with profiler.record_function("mppi/reset_cov"):
            # init_cov can either be a single value, or n_problems x 1 or n_problems x d_action

            if self.cov_type == CovType.SIGMA_I:
                # init_cov can either be a single value, or n_problems x 1
                self.cov_action = self.init_cov
                if self.init_cov.shape[0] != self.n_problems:
                    self.cov_action = self.init_cov.unsqueeze(0).expand(self.n_problems, -1)
                self.inv_cov_action = 1.0 / self.cov_action
                a = torch.sqrt(self.cov_action)
                if not copy_tensor(a, self.scale_tril):
                    self.scale_tril = a

            elif self.cov_type == CovType.DIAG_A:
                # init_cov can either be a single value, or n_problems x 1 or n_problems x 7
                init_cov = self.init_cov.clone()

                # if(init_cov.shape[-1] != self.d_action):
                if len(init_cov.shape) == 1:
                    init_cov = init_cov.unsqueeze(-1).expand(-1, self.d_action)
                if len(init_cov.shape) == 2 and init_cov.shape[-1] != self.d_action:
                    init_cov = init_cov.expand(-1, self.d_action)
                init_cov = init_cov.unsqueeze(1)
                if init_cov.shape[0] != self.n_problems:
                    init_cov = init_cov.expand(self.n_problems, -1, -1)
                if not copy_tensor(init_cov.clone(), self.cov_action):
                    self.cov_action = init_cov.clone()
                self.inv_cov_action = 1.0 / self.cov_action
                a = torch.sqrt(self.cov_action)
                if not copy_tensor(a, self.scale_tril):
                    self.scale_tril = a

            else:
                raise ValueError("Unidentified covariance type in update_distribution")

    def _get_action_seq(self, mode: SampleMode):
        if mode == SampleMode.MEAN:
            act_seq = self.mean_action  # .clone()  # [self.mean_idx]#.clone()
        elif mode == SampleMode.SAMPLE:
            delta = self.generate_noise(
                shape=torch.Size((1, self.action_horizon)),
                base_seed=self.seed + 123 * self.num_steps,
            )
            act_seq = self.mean_action + torch.matmul(delta, self.full_scale_tril)
        elif mode == SampleMode.BEST:
            act_seq = self.best_traj  # [self.mean_idx]
        else:
            raise ValueError("Unidentified sampling mode in get_next_action")

        # act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs, squash_fn=self.squash_fn)

        return act_seq

    def generate_noise(self, shape, base_seed=None):
        """
        Generate correlated noisy samples using autoregressive process
        """
        delta = self.sample_lib.get_samples(sample_shape=shape, seed=base_seed)
        return delta

    @property
    def full_scale_tril(self):
        """Returns the full scale tril

        Returns:
            Tensor: dimension is (d_action, d_action)
        """
        if self.cov_type == CovType.SIGMA_I:
            return (
                self.scale_tril.unsqueeze(-2).unsqueeze(-2).expand(-1, -1, self.action_horizon, -1)
            )
        elif self.cov_type == CovType.DIAG_A:
            return self.scale_tril.unsqueeze(-2).expand(-1, -1, self.action_horizon, -1)  # .cl
        elif self.cov_type == CovType.FULL_A:
            return self.scale_tril
        elif self.cov_type == CovType.FULL_HA:
            return self.scale_tril

    def _calc_val(self, trajectories: Trajectory):
        costs = trajectories.costs
        actions = trajectories.actions
        delta = actions - self.mean_action.unsqueeze(0)

        traj_costs = cost_to_go(costs, self.gamma_seq)[:, 0]
        control_costs = self._control_costs(delta)
        total_costs = traj_costs + self.beta * control_costs

        val = -self.beta * torch.logsumexp((-1.0 / self.beta) * total_costs)
        return val

    def reset(self):
        self.reset_distribution()

        self._sample_iter[:] = 0
        self._sample_iter_n = 0
        self.update_samples()  # this helps in restarting optimization
        super().reset()

    @property
    def squashed_mean(self):
        return scale_ctrl(
            self.mean_action, self.action_lows, self.action_highs, squash_fn=self.squash_fn
        )

    @property
    def full_cov(self):
        if self.cov_type == CovType.SIGMA_I:
            return self.cov_action * self.I
        elif self.cov_type == CovType.DIAG_A:
            return torch.diag(self.cov_action)
        elif self.cov_type == CovType.FULL_A:
            return self.cov_action
        elif self.cov_type == CovType.FULL_HA:
            return self.cov_action

    @property
    def full_inv_cov(self):
        if self.cov_type == CovType.SIGMA_I:
            return self.inv_cov_action * self.I
        elif self.cov_type == CovType.DIAG_A:
            return torch.diag_embed(self.inv_cov_action)
        elif self.cov_type == CovType.FULL_A:
            return self.inv_cov_action
        elif self.cov_type == CovType.FULL_HA:
            return self.inv_cov_action

    @property
    def full_scale_tril(self):
        if self.cov_type == CovType.SIGMA_I:
            return (
                self.scale_tril.unsqueeze(-2).unsqueeze(-2).expand(-1, -1, self.action_horizon, -1)
            )  # .cl
        elif self.cov_type == CovType.DIAG_A:
            return self.scale_tril.unsqueeze(-2).expand(-1, -1, self.action_horizon, -1)  # .cl
        elif self.cov_type == CovType.FULL_A:
            return self.scale_tril
        elif self.cov_type == CovType.FULL_HA:
            return self.scale_tril

    @property
    def entropy(self):
        ent_L = gaussian_entropy(L=self.full_scale_tril)
        return ent_L

    def reset_seed(self):
        self.sample_lib = SampleLib(self.sample_params)
        self.mean_lib = HaltonSampleLib(
            SampleConfig(
                self.action_horizon,
                self.d_action,
                tensor_args=self.tensor_args,
                **{"fixed_samples": False, "seed": 2567, "filter_coeffs": None}
            )
        )
        # resample if not fixed samples:
        self.update_samples()
        super().reset_seed()

    def update_samples(self):
        with profiler.record_function("mppi/update_samples"):
            if self.sample_params.fixed_samples:
                n_iters = 1
            else:
                n_iters = self.n_iters
            if self.sample_per_problem:
                s_set = (
                    self.sample_lib.get_samples(
                        sample_shape=[
                            self.sampled_particles_per_problem * self.n_problems * n_iters
                        ],
                        base_seed=self.seed,
                    )
                    .view(
                        n_iters,
                        self.n_problems,
                        self.sampled_particles_per_problem,
                        self.action_horizon,
                        self.d_action,
                    )
                    .clone()
                )
            else:
                s_set = self.sample_lib.get_samples(
                    sample_shape=[n_iters * (self.sampled_particles_per_problem)],
                    base_seed=self.seed,
                )
                s_set = s_set.view(
                    n_iters,
                    1,
                    self.sampled_particles_per_problem,
                    self.action_horizon,
                    self.d_action,
                )
                s_set = s_set.repeat(1, self.n_problems, 1, 1, 1).clone()
            s_set[:, :, -1, :, :] = 0.0
            if not copy_tensor(s_set, self._sample_set):
                log_info("ParallelMPPI: Updating sample set")
                self._sample_set = s_set
            if self._sample_iter is None:
                log_info("ParallelMPPI: Resetting sample iterations")  # , sample_iter.shape)

                self._sample_iter = torch.zeros(
                    (1), dtype=torch.long, device=self.tensor_args.device
                )
            else:
                self._sample_iter[:] = 0

            # if not copy_tensor(sample_iter, self._sample_iter):
            #    log_info("ParallelMPPI: Resetting sample iterations")  # , sample_iter.shape)
            #    self._sample_iter = sample_iter
            self._sample_iter_n = 0

    @torch.no_grad()
    def generate_rollouts(self, init_act=None):
        """
        Samples a batch of actions, rolls out trajectories for each particle
        and returns the resulting observations, costs,
        actions

        Parameters
        ----------
        state : dict or np.ndarray
            Initial state to set the simulation problem to
        """

        return super().generate_rollouts(init_act)


@get_torch_jit_decorator()
def jit_calculate_exp_util(beta: float, total_costs):
    w = torch.softmax((-1.0 / beta) * total_costs, dim=-1)
    return w


# @get_torch_jit_decorator()
def jit_calculate_exp_util_from_costs(costs, gamma_seq, beta: float): #, col_mask:Optional[torch.Tensor]=None):
    
    # discounted costs (by gamma)
    cost_seq = gamma_seq * costs
    cost_seq = torch.sum(cost_seq, dim=-1, keepdim=False) / gamma_seq[..., 0] #[B x Nrollouts] discounted cost sum over horizon and mean across particles

    # if col_mask is not None: # Upper bound mode
    #     # print("DEBUG: Upper bound mode")
    #     cfree_rollout_mask = (torch.sum(col_mask, dim=-1) == 0).unsqueeze(0) # 1 where no collision in rollout, 0 elsewhere
    #     col_rollout_mask = (torch.sum(col_mask, dim=-1) > 0).unsqueeze(0) # 1 where collision in rollout, 0 elsewhere
    #     cfree_rollouts = cost_seq[cfree_rollout_mask]
    #     col_rollouts = cost_seq[col_rollout_mask]
        
    #     # print(f'DEBUG: len(cfree_rollouts): {len(cfree_rollouts)}, len(col_rollouts): {len(col_rollouts)}')
    #     if len(cfree_rollouts) > 0 and len(col_rollouts) > 0:
    #         max_cfree_cost = torch.max(cfree_rollouts)
    #         min_col_cost = torch.min(col_rollouts)
    #         # print(f"DEBUG: max_cfree_cost: {max_cfree_cost}, min_col_cost: {min_col_cost}")
    #         k = torch.max(torch.tensor(0.0), max_cfree_cost - min_col_cost) + 1000 
    #         # print(f"DEBUG adding k: {k} to col rollouts: (N_col_rollouts: {len(col_rollouts)}), N_cfree_rollouts: {len(cfree_rollouts)}")
    #         cost_seq[col_rollout_mask] = cost_seq[col_rollout_mask] + k # add k to col rollouts to make them larger than col free rollouts
            
    
    w = torch.softmax((-1.0 / beta) * cost_seq, dim=-1)
    return w


@get_torch_jit_decorator()
def jit_compute_total_cost(gamma_seq, costs):
    cost_seq = gamma_seq * costs
    cost_seq = torch.sum(cost_seq, dim=-1, keepdim=False) / gamma_seq[..., 0]
    return cost_seq


@get_torch_jit_decorator()
def jit_diag_a_cov_update(w, actions, mean_action):
    delta_actions = actions - mean_action.unsqueeze(-3)

    weighted_delta = w * (delta_actions**2)
    # weighted_delta =
    # sum across horizon and mean across particles:
    # cov_update = torch.diag(torch.mean(torch.sum(weighted_delta.T  , dim=0), dim=0))
    cov_update = torch.mean(torch.sum(weighted_delta, dim=-3), dim=-2).unsqueeze(-2)
    return cov_update


@get_torch_jit_decorator()
def jit_blend_cov(cov_action, cov_update, step_size_cov: float, kappa: float):
    new_cov = (1.0 - step_size_cov) * cov_action + step_size_cov * cov_update + kappa
    return new_cov


@get_torch_jit_decorator()
def jit_blend_mean(mean_action, new_mean, step_size_mean: float):
    mean_update = (1.0 - step_size_mean) * mean_action + step_size_mean * new_mean
    return mean_update


@get_torch_jit_decorator()
def jit_mean_cov_diag_a(
    costs,
    actions,
    gamma_seq,
    mean_action,
    cov_action,
    step_size_mean: float,
    step_size_cov: float,
    kappa: float,
    beta: float,
):
    w = jit_calculate_exp_util_from_costs(costs, gamma_seq, beta)
    w = w.unsqueeze(-1).unsqueeze(-1)
    new_mean = torch.sum(w * actions, dim=-3)
    new_mean = jit_blend_mean(mean_action, new_mean, step_size_mean)
    cov_update = jit_diag_a_cov_update(w, actions, mean_action)
    new_cov = jit_blend_cov(cov_action, cov_update, step_size_cov, kappa)
    new_tril = torch.sqrt(new_cov)
    return new_mean, new_cov, new_tril