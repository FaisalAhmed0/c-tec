import gym
from typing import Dict, Any

import numpy as np
from gym import spaces
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from src.algo.intrinsic_rewards.base_model import IntrinsicRewardBaseModel
from src.algo.common_models.mlps import *
from src.utils.enum_types import NormType
from src.utils.common_func import init_module_with_name
from src.utils.running_mean_std import RunningMeanStd
import pdb


'''
1. Add the l1-distance
2. modify the intrinisc reward function 
3. add the learned temparture
'''

def mrn_distance(x, y):
    # Metric Residual Network (MRN) architecture (https://arxiv.org/pdf/2208.08133)
    eps = 1e-8
    d = x.shape[-1]
    x_prefix = x[..., :d // 2]
    x_suffix = x[..., d // 2:]
    y_prefix = y[..., :d // 2]
    y_suffix = y[..., d // 2:]
    max_component = th.max(F.relu(x_prefix - y_prefix), axis=-1).values
    l2_component = th.sqrt(th.square(x_suffix - y_suffix).sum(axis=-1) + eps)
    return max_component + l2_component
    

class CTeCModel(IntrinsicRewardBaseModel):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        max_grad_norm: float = 0.5,
        model_learning_rate: float = 3e-4,
        model_cnn_features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        model_cnn_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        model_features_dim: int = 256,
        model_latents_dim: int = 256,
        model_mlp_norm: NormType = NormType.BatchNorm,
        model_cnn_norm: NormType = NormType.BatchNorm,
        model_gru_norm: NormType = NormType.NoNorm,
        use_model_rnn: int = 0,
        model_mlp_layers: int = 1,
        gru_layers: int = 1,
        use_status_predictor: int = 0,
        tdd_aggregate_fn: bool = 'min',
        ctec_energy_fn: str = 'l1', # I changed this to l1
        ctec_loss_fn: str = 'infonce',
        ctec_logsumexp_coef: float = 0.1,
        normalize_repr: int = 1,
        offpolicy_data: int = 0,
        auto_temp: int = 1,
        use_etd_rwd: int = 0
    ):
        super().__init__(observation_space, action_space, activation_fn, normalize_images,
                         optimizer_class, optimizer_kwargs, max_grad_norm, model_learning_rate,
                         model_cnn_features_extractor_class, model_cnn_features_extractor_kwargs,
                         model_features_dim, model_latents_dim, model_mlp_norm,
                         model_cnn_norm, model_gru_norm, use_model_rnn, model_mlp_layers,
                         gru_layers, use_status_predictor)
        self.aggregate_fn = tdd_aggregate_fn # [min, quantile10, knn10]
        self.energy_fn = ctec_energy_fn    # [l2, mrn, mrn_pot, cos]
        self.loss_fn = ctec_loss_fn           # [infonce, infonce_symmetric, infonce_backward]
        self.offpolicy_data = offpolicy_data #  [0, 1]
        self.auto_temp = auto_temp
        if auto_temp:
            self.log_temp = nn.Parameter(th.tensor(0.0, dtype=th.float32))
        self.temperature = 1.
        self.normalize_repr = normalize_repr
        self.knn_k = 10
        self.logsumexp_coef = ctec_logsumexp_coef
        self.use_etd_rwd = use_etd_rwd
        self._build()
        self._init_modules()
        self._init_optimizers()

        # print("C-TeC Models")
        # print(self.model_cnn_extractor)
        # print(self.curr_state_action_encoder)
        # print(self.future_state_encoder)
        # pdb.set_trace()

    def _init_modules(self) -> None:
        module_names = {
            self.model_cnn_extractor: 'model_cnn_extractor',
            self.curr_state_action_encoder: 'curr_state_action_encoder',
            self.future_state_encoder: 'future_state_encoder',
        }
        for module, name in module_names.items():
            init_module_with_name(name, module)

    def _init_optimizers(self) -> None:
        param_dicts = dict(self.named_parameters(recurse=True)).items()
        self.model_params = [
            param for name, param in param_dicts
        ]
        self.model_optimizer = self.optimizer_class(self.model_params, lr=self.model_learning_rate, **self.optimizer_kwargs)
    
    def _build(self) -> None:
        # Build CNN
        self.model_cnn_features_extractor_kwargs.update(dict(
            features_dim=self.model_features_dim,
        ))
        self.model_cnn_extractor = \
            self.model_cnn_features_extractor_class(
                self.observation_space,
                **self.model_cnn_features_extractor_kwargs
            )
        self.curr_state_action_encoder = ModelOutputHeads(
            feature_dim=self.model_features_dim + self.action_num,
            latent_dim=self.model_latents_dim,
            activation_fn=self.activation_fn,
            mlp_norm=self.model_mlp_norm,
            mlp_layers=self.model_mlp_layers,
            output_dim=64,
        )
        self.future_state_encoder = ModelOutputHeads(
            feature_dim=self.model_features_dim,
            latent_dim=self.model_latents_dim,
            activation_fn=self.activation_fn,
            mlp_norm=self.model_mlp_norm,
            mlp_layers=self.model_mlp_layers,
            output_dim=64,
        )
        # self.potential_net = ModelOutputHeads(
        #     feature_dim=self.model_features_dim,
        #     latent_dim=self.model_latents_dim,
        #     activation_fn=self.activation_fn,
        #     mlp_norm=self.model_mlp_norm,
        #     mlp_layers=self.model_mlp_layers,
        #     output_dim=1,
        # )

    def forward(self,
        curr_obs: Tensor, future_obs: Tensor, curr_action: Tensor
    ):
        # CNN Extractor
        # print("in C-tec forward method")
        # pdb.set_trace()
        batch_size = curr_obs.size(0)
        # import pdb;pdb.set_trace()
        # if len(curr_obs.shape) > 3:
        #     curr_obs = curr_obs.reshape(curr_obs.shape[0], 3, curr_obs.shape[-2], curr_obs.shape[-1])
        #     future_obs = future_obs.reshape(future_obs.shape[0]*future_obs.shape[1], 3, future_obs.shape[-2], future_obs.shape[-1])
        curr_cnn_embs = self._get_cnn_embeddings(curr_obs).view(batch_size, -1)
        next_cnn_embs = self._get_cnn_embeddings(future_obs).view(batch_size, -1)
        curr_action = curr_action.view(batch_size, -1)
        if curr_action.shape[-1] == 1:
            curr_action = F.one_hot(curr_action.long().squeeze(),num_classes=self.action_num)
        phi_x = self.curr_state_action_encoder( th.cat([curr_cnn_embs, curr_action], axis=-1) )
        phi_y = self.future_state_encoder(next_cnn_embs)
        device = phi_x.device

        if self.normalize_repr or self.energy_fn == "cos":
            phi_x_norm = th.linalg.norm(phi_x, axis=-1, keepdims=True)
            phi_y_norm = th.linalg.norm(phi_y, axis=-1, keepdims=True)
            phi_x = phi_x/phi_x_norm
            phi_y = phi_y/phi_y_norm
            if self.auto_temp:
                phi_x = phi_x / th.exp(self.log_temp)
            else:
                phi_x = phi_x / self.temperature

        if self.energy_fn == 'l2':
            logits = - th.sqrt(((phi_x[:, None] - phi_y[None, :])**2).sum(dim=-1) + 1e-8)
        elif self.energy_fn == "l1":
            logits = - th.abs(phi_x[:, None] - phi_y[None, :]).sum(dim=-1) + 1e-8
        elif self.energy_fn == 'cos':
            logits = th.einsum("ik,jk->ij", phi_x, phi_y)
        elif self.energy_fn == 'dot':
            logits = th.einsum("ik,jk->ij", phi_x, phi_y)
        elif self.energy_fn == 'mrn':
            logits = - mrn_distance(phi_x[:, None], phi_y[None, :])
        # elif self.energy_fn == 'mrn_pot':
        #     logits = c_y.T - mrn_distance(phi_x[:, None], phi_y[None, :])
        
        batch_size = logits.size(0)
        I = th.eye(batch_size, device=device)
        if self.loss_fn == 'infonce':
            contrastive_loss = F.cross_entropy(logits, I)
        elif self.loss_fn == 'infonce_backward':
            contrastive_loss = F.cross_entropy(logits.T, I)
        elif self.loss_fn == 'infonce_symmetric':
            contrastive_loss = (F.cross_entropy(logits, I) + F.cross_entropy(logits.T, I)) / 2
        elif self.loss_fn == 'dpo':
            positive = th.diag(logits)
            diffs = positive[:, None] - logits
            contrastive_loss = -F.logsigmoid(diffs)

        contrastive_loss = th.mean(contrastive_loss)
        
        # Log
        logs = {
            'contrastive_loss': contrastive_loss,
            'logits_pos': th.diag(logits).mean(), 
            'logits_neg': th.mean(logits * (1 - I)),
            'logits_logsumexp': th.mean((th.logsumexp(logits + 1e-6, axis=1)**2)),
            'categorical_accuracy': th.mean((th.argmax(logits, axis=1) == th.arange(batch_size, device=device)).float()),
        }
        return contrastive_loss, logs

    def get_intrinsic_rewards(self,
        curr_obs, future_obs , next_obs, last_mems, curr_act, curr_dones, obs_history, stats_logger
    ):
        with th.no_grad():
            # CNN Extractor
            batch_size = curr_obs.size(0)
            num_envs = curr_obs.size(1)
            if len(curr_obs.shape) > 3:
                curr_obs = curr_obs.reshape(curr_obs.shape[0]*curr_obs.shape[1], 3, curr_obs.shape[-2], curr_obs.shape[-1])
                future_obs = future_obs.reshape(future_obs.shape[0]*future_obs.shape[1], 3, future_obs.shape[-2], future_obs.shape[-1])
            curr_obs_embs = self._get_cnn_embeddings(curr_obs).view(batch_size*num_envs, -1)
            future_obs_embs = self._get_cnn_embeddings(future_obs).view(batch_size*num_envs, -1)
            curr_act = curr_act.view(batch_size*num_envs, -1)
            if curr_act.shape[-1] == 1:
                curr_act = F.one_hot(curr_act.squeeze(),num_classes=self.action_num)

            phi_x = self.curr_state_action_encoder(th.cat([curr_obs_embs, curr_act], axis=-1))
            phi_y = self.future_state_encoder(future_obs_embs)

            if self.normalize_repr or self.energy_fn == "cos":
                phi_x_norm = th.linalg.norm(phi_x, axis=-1, keepdims=True)
                phi_y_norm = th.linalg.norm(phi_y, axis=-1, keepdims=True)
                phi_x = phi_x/phi_x_norm
                phi_y = phi_y/phi_y_norm
                if self.auto_temp:
                    phi_x = phi_x / th.exp(self.log_temp)
                else:
                    phi_x = phi_x / self.temperature

            # Compute dists
            if self.energy_fn == 'l2':
                dists = th.sqrt(((phi_x[:, None] - phi_y[None, :])**2).sum(dim=-1) + 1e-8)
            elif self.energy_fn == "l1":
                dists = th.abs(phi_x[:, None] - phi_y[None, :]).sum(dim=-1) + 1e-8
            elif self.energy_fn == 'cos':
                x_norm = th.linalg.norm(phi_x, axis=-1, keepdims=True)
                y_norm = th.linalg.norm(phi_y, axis=-1, keepdims=True)
                phi_x_norm = phi_x / x_norm
                phi_y_norm = phi_y / y_norm
                phi_x_norm = phi_x_norm / self.temperature
                dists = th.einsum("ik,jk->ij", phi_x_norm, phi_y_norm)
            elif self.energy_fn == 'dot':
                dists = th.einsum("ik,jk->ij", phi_x, phi_y)
            elif self.energy_fn == 'mrn':
                dists = mrn_distance(phi_x[:, None], phi_y[None, :])
            int_rews = dists.diag()
                        
        logs = {
            'dists_mean': dists.mean(),
            'dists_min': dists.min(),
            'dists_max': dists.max(),
        }
        stats_logger.add(
            **logs,
        )
        # import pdb;pdb.set_trace()
        int_rews = int_rews.view(batch_size, num_envs)
        return int_rews, None

    def get_etd_intrinsic_reward(self,
        curr_obs, next_obs, next_action, last_mems,curr_dones, obs_history, stats_logger):
        with th.no_grad():
            # CNN Extractor
            batch_size = curr_obs.size(0)
            curr_cnn_embs = self._get_cnn_embeddings(curr_obs)
            next_cnn_embs = self._get_cnn_embeddings(next_obs)

            
            int_rews = np.zeros(batch_size, dtype=np.float32)
            for env_id in range(batch_size):
                # Update historical observation embeddings
                curr_obs_emb = curr_cnn_embs[env_id].view(1, -1)
                next_obs_emb = next_cnn_embs[env_id].view(1, -1)
                obs_embs = obs_history[env_id]
                new_embs = [curr_obs_emb, next_obs_emb] if obs_embs is None else [obs_embs, next_obs_emb]
                obs_embs = th.cat(new_embs, dim=0)
                obs_history[env_id] = obs_embs
                phi_x = self.future_state_encoder(obs_history[env_id][:-1])
                if len(next_action.shape) == 1:
                    next_action_emb = F.one_hot(next_action[env_id].long(), num_classes=self.action_num).unsqueeze(0)
                else:
                    next_action_emb = next_action[env_id].unsqueeze(0)
                phi_y = self.curr_state_action_encoder(th.cat([obs_history[env_id][-1].unsqueeze(0), next_action_emb],dim=-1 ))
                # Compute dists
                if self.energy_fn == 'l2':
                    dists = th.sqrt(((phi_x[:, None] - phi_y[None, :])**2).sum(dim=-1) + 1e-8)
                elif self.energy_fn == 'l1':
                    dists = th.abs(((phi_x[:, None] - phi_y[None, :])).sum(dim=-1) + 1e-8)
                elif self.energy_fn == 'cos':
                    x_norm = th.linalg.norm(phi_x, axis=-1, keepdims=True)
                    y_norm = th.linalg.norm(phi_y, axis=-1, keepdims=True)
                    phi_x_norm = phi_x / x_norm
                    phi_y_norm = phi_y / y_norm
                    phi_x_norm = phi_x_norm / self.temperature
                    dists = - th.einsum("ik,jk->ij", phi_x_norm, phi_y_norm)
                elif self.energy_fn == 'dot':
                    dists = - th.einsum("ik,jk->ij", phi_x, phi_y)
                elif 'mrn' in self.energy_fn:
                    dists = mrn_distance(phi_x, phi_y)
                # Compute intrinsic reward
                if self.aggregate_fn == 'min':
                    int_rew = dists.min().item()
                    int_rews[env_id] += int_rew
                elif self.aggregate_fn == 'quantile10':
                    int_rews[env_id] += th.quantile(dists, 0.1).item()
                elif self.aggregate_fn == 'knn':
                    if len(dists) <= self.knn_k:
                        knn_dists = dists
                    else:
                        knn_dists, _ = th.topk(dists, self.knn_k, largest=False)
                    int_rews[env_id] += knn_dists[-1].item()
                        
        logs = {
            'dists_mean': dists.mean(),
            'dists_min': dists.min(),
            'dists_max': dists.max(),
        }
        stats_logger.add(
            **logs,
        )
        return int_rews, None



    def optimize(self, rollout_data, offpolicy_data, stats_logger):
        # print("In C-TeC optimize method")
        # pdb.set_trace()
        if self.offpolicy_data:
            contrastive_loss, logs = \
                self.forward(
                    offpolicy_data.observations,
                    offpolicy_data.future_observations,
                    offpolicy_data.actions
                )
        else:
            contrastive_loss, logs = \
                self.forward(
                    rollout_data.observations,
                    rollout_data.future_observations,
                    rollout_data.actions
                )
        loss = contrastive_loss + self.logsumexp_coef * logs['logits_logsumexp']
        self.model_optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.model_params, self.max_grad_norm)
        self.model_optimizer.step()

        stats_logger.add(
            **logs,
        )