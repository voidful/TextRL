import itertools

import pfrl
import torch
import torch.nn.functional as F
from pfrl.agents.ppo import _elementwise_clip
from torch import autocast


class TextRLActor:
    @autocast('cuda')
    def __init__(self, env, model, tokenizer, device=0, act_deterministically=True):
        self.agent = None
        self.n_actions = max(model.config.vocab_size, tokenizer.vocab_size)
        self.env = env
        self.device = device
        self.model = model
        self.obs_size = model.config.hidden_size
        self.converter = self.model.lm_head
        self.act_deterministically = act_deterministically

    @autocast('cuda')
    def agent_ppo(self, update_interval=10, minibatch_size=3000, epochs=20):
        policy = torch.nn.Sequential(
            self.converter,
            SoftmaxCategoricalHead()
        )
        vf = torch.nn.Sequential(
            torch.nn.Linear(self.obs_size, 1),
        )
        model = pfrl.nn.Branched(policy, vf)
        opt = torch.optim.SGD(model.parameters(), lr=5e-5)
        model = model.cuda()
        agent = TextPPO(
            model,
            opt,
            gpu=self.device,
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
    def predict(self, input_item, max_episode_len=100):
        t = 0
        with torch.inference_mode():
            with self.agent.eval_mode():
                obs = self.env.reset(input_item)
                while True:
                    action = self.agent.act(obs)
                    obs, reward, done, pred = self.env.step(action)
                    t += 1
                    reset = t >= max_episode_len
                    self.agent.observe(obs, reward, done, reset)
                    if done or reset:
                        return pred.get('predicted_str')


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


class SoftmaxCategoricalHead(torch.nn.Module):
    @autocast('cuda')
    def forward(self, logits, temperature=0.3):
        softmax = torch.nn.Softmax(dim=1)
        return torch.distributions.Categorical(probs=softmax(logits / temperature))
