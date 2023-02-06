import itertools

import pfrl
import torch
import torch.nn.functional as F
from pfrl.agents.ppo import _elementwise_clip
from torch import autocast
from transformers import top_k_top_p_filtering


class TextRLActor:
    @autocast('cuda')
    def __init__(self, env, model, tokenizer, gpu_id=0, act_deterministically=True,
                 temperature=0.6,
                 compare_sample=3,
                 top_k=0,
                 top_p=1.0,
                 repetition_penalty=1.0):
        self.agent = None
        self.n_actions = max(model.config.vocab_size, tokenizer.vocab_size)
        self.env = env
        self.gpu_id = gpu_id
        self.device = torch.device("cuda:{}".format(gpu_id))
        self.model = model
        self.obs_size = model.config.hidden_size
        self.converter = self.model.lm_head
        self.act_deterministically = act_deterministically
        self.temperature = temperature
        self.compare_sample = compare_sample * 2
        self.top_k = top_k
        self.tok_p = top_p
        self.repetition_penalty = repetition_penalty

    @autocast('cuda')
    def agent_ppo(self, update_interval=10, minibatch_size=3000, epochs=20, lr=5e-5):
        policy = torch.nn.Sequential(
            self.converter,
            torch.nn.Flatten(start_dim=0, end_dim=1),
            SoftmaxCategoricalHead(self.env,
                                   temperature=self.temperature,
                                   compare_sample=self.compare_sample,
                                   top_k=self.top_k,
                                   top_p=self.tok_p,
                                   repetition_penalty=self.repetition_penalty)
        )
        vf = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=0, end_dim=1),
            torch.nn.Linear(self.obs_size, 1),
        )
        model = pfrl.nn.Branched(policy, vf)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        model = model.cuda()
        agent = TextPPO(
            model,
            opt,
            gpu=self.gpu_id,
            update_interval=update_interval,
            minibatch_size=minibatch_size,
            epochs=epochs,
            clip_eps_vf=None,
            entropy_coef=0,
            gamma=1,
            lambd=1,
            standardize_advantages=True,
            act_deterministically=self.act_deterministically
        )
        self.agent = agent
        return agent

    @autocast('cuda')
    def predict(self, input_item):
        t = 0
        with torch.inference_mode():
            with self.agent.eval_mode():
                obs = self.env.reset(input_item)
                while True:
                    action = self.agent.act(obs)
                    obs, reward, done, pred = self.env.step(action)
                    t += 1
                    reset = t >= self.env.env_max_length
                    self.agent.observe(obs, reward, done, reset)
                    if done or reset:
                        return pred.get('predicted_str')


class SoftmaxCategoricalHead(torch.nn.Module):
    def __init__(self, env, temperature=0.6, compare_sample=3, top_k=0, top_p=1.0, repetition_penalty=1.0):
        super().__init__()
        self.env = env
        self.softmax = torch.nn.Softmax(dim=-1)
        self.temperature = temperature
        self.compare_sample = compare_sample
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    @autocast('cuda')
    def forward(self, logits):
        # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
        # repetition penalty from https://github.com/huggingface/transformers/pull/2303/files#diff-6b72b98c4c2dcfc6cc606843917733f5d858374fbc22a735ff483bbc0c1e63ea
        if self.repetition_penalty != 1.0:
            for seq_num, predicted in enumerate(self.env.predicted):
                for previous_tokens in set(predicted):
                    prev_token_id = self.env.tokenizer.convert_tokens_to_ids(previous_tokens)
                    # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    if logits[seq_num, prev_token_id] < 0:
                        logits[seq_num, prev_token_id] *= self.repetition_penalty
                    else:
                        logits[seq_num, prev_token_id] /= self.repetition_penalty
        logits = logits / self.temperature
        logits = top_k_top_p_filtering(logits, top_k=self.top_k, top_p=self.top_p)
        return torch.distributions.Multinomial(total_count=self.compare_sample, probs=self.softmax(logits))


class TextPPO(pfrl.agents.PPO):
    @autocast('cuda')
    def _update_if_dataset_is_ready(self):
        dataset_size = (
                sum(len(episode) for episode in self.memory)
                + len(self.last_episode)
                + (
                    0
                    if self.batch_last_episode is None
                    else sum(len(episode) for episode in self.batch_last_episode)
                )
        )
        if dataset_size >= self.update_interval:
            self._flush_last_episode()
            if self.recurrent:
                dataset = pfrl.agents.ppo._make_dataset_recurrent(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    max_recurrent_sequence_len=self.max_recurrent_sequence_len,
                    device=self.device,
                )
                self._update_recurrent(dataset)
            else:
                dataset = pfrl.agents.ppo._make_dataset(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    device=self.device,
                )
                assert len(dataset) == dataset_size
                self._update(dataset)
            self.explained_variance = pfrl.agents.ppo._compute_explained_variance(
                list(itertools.chain.from_iterable(self.memory))
            )
            self.memory = []

    @autocast('cuda')
    def batch_act(self, batch_obs):
        if self.training:
            return self._batch_act_train(batch_obs)
        else:
            return self._batch_act_eval(batch_obs)

    @autocast('cuda')
    def _lossfun(
            self, entropy, vs_pred, log_probs, vs_pred_old, log_probs_old, advs, vs_teacher
    ):

        prob_ratio = torch.exp(log_probs - log_probs_old)
        loss_policy = -torch.mean(
            torch.min(
                (prob_ratio * advs),
                (torch.clamp(prob_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs),
            ),
        )

        if self.clip_eps_vf is None:
            loss_value_func = F.mse_loss(vs_pred, vs_teacher)
        else:
            clipped_vs_pred = _elementwise_clip(
                vs_pred,
                vs_pred_old - self.clip_eps_vf,
                vs_pred_old + self.clip_eps_vf,
            )
            loss_value_func = torch.mean(
                torch.max(
                    F.mse_loss(vs_pred, vs_teacher, reduction="none"),
                    F.mse_loss(clipped_vs_pred, vs_teacher, reduction="none"),
                )
            )
        loss_entropy = -torch.mean(entropy)

        self.value_loss_record.append(float(loss_value_func))
        self.policy_loss_record.append(float(loss_policy))

        loss = (
                loss_policy
                + self.value_func_coef * loss_value_func
                + self.entropy_coef * loss_entropy
        )
        return loss
