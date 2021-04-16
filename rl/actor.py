import copy

import pfrl
import torch


class TextRLActor:
    def __init__(self, env, model, tokenizer, device=0):
        self.agent = None
        self.n_actions = tokenizer.vocab_size
        self.env = env
        self.device = device
        self.model = model
        self.obs_size = model.config.hidden_size
        self.converter = torch.nn.Linear(self.obs_size, self.n_actions)
        self.converter.weight = copy.deepcopy(self.model.lm_head.weight)

    def agent_ppo(self, update_interval=10, minibatch_size=3000, epochs=20):
        policy = torch.nn.Sequential(
            self.converter,
            pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                action_size=self.n_actions,
                var_type="diagonal",
                var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            ),
        )
        vf = torch.nn.Sequential(
            torch.nn.Linear(self.obs_size, 1),
        )
        model = pfrl.nn.Branched(policy, vf)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        agent = pfrl.agents.PPO(
            model,
            opt,
            gpu=self.device,
            update_interval=update_interval,
            minibatch_size=minibatch_size,
            epochs=epochs,
            clip_eps_vf=None,
            entropy_coef=0,
            standardize_advantages=True,
            gamma=0.995,
            lambd=0.97,
            act_deterministically=True
        )
        self.agent = agent
        return agent

    def predict(self, input_text, max_episode_len=100):
        t = 0
        with self.agent.eval_mode():
            obs = self.env.reset(input_text)
            while True:
                action = self.agent.act(obs)
                obs, reward, done, pred = self.env.step(action)
                t += 1
                reset = t >= max_episode_len
                self.agent.observe(obs, reward, done, reset)
                if done or reset:
                    return pred.get('predicted_str')
