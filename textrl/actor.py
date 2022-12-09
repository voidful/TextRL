import copy

import pfrl
import torch
import torch.nn.functional as F
from pfrl.agents.ppo import _elementwise_clip


class TextRLActor:
    def __init__(self, env, model, tokenizer, device=0):
        self.agent = None
        self.n_actions = max(model.config.vocab_size, tokenizer.vocab_size)
        self.env = env
        self.device = device
        self.model = model
        self.obs_size = model.config.hidden_size
        self.converter = torch.nn.Linear(self.obs_size, self.n_actions)
        self.converter.weight = copy.deepcopy(self.model.lm_head.weight)

    def agent_ppo(self, update_interval=10, minibatch_size=3000, epochs=20):
        policy = torch.nn.Sequential(
            self.converter,
            SoftmaxCategoricalHead()
        )
        vf = torch.nn.Sequential(
            torch.nn.Linear(self.obs_size, 1),
        )
        model = pfrl.nn.Branched(policy, vf)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
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
            act_deterministically=True
        )
        self.agent = agent
        return agent

    def predict(self, input_item, max_episode_len=100):
        t = 0
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
    def _lossfun(
            self, entropy, vs_pred, log_probs, vs_pred_old, log_probs_old, advs, vs_teacher
    ):

        prob_ratio = torch.exp(log_probs - log_probs_old)
        loss_policy = -torch.mean(
            torch.min(
                (prob_ratio.T * advs).T,
                (torch.clamp(prob_ratio, 1 - self.clip_eps, 1 + self.clip_eps).T * advs).T,
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
    def forward(self, logits, temperature=0.1):
        softmax = torch.nn.Softmax(dim=1)
        return torch.distributions.Categorical(probs=softmax(logits / temperature))