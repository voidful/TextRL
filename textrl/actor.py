import copy
import itertools

import pfrl
import torch
import torch.nn.functional as F
from pfrl.agents.ppo import _elementwise_clip
from pfrl.utils.mode_of_distribution import mode_of_distribution
from torch import autocast


class TextRLActor:
    @autocast('cuda')
    def __init__(self, env, model, tokenizer, gpu_id=0, act_deterministically=True,
                 temperature=0.6,
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
        # self.original_weight = copy.deepcopy(self.model.lm_head.weight)
        self.act_deterministically = act_deterministically
        self.temperature = temperature
        self.top_k = top_k
        self.tok_p = top_p
        self.repetition_penalty = repetition_penalty

    @autocast('cuda')
    def agent_ppo(self, update_interval=10, minibatch_size=3000, epochs=20, lr=5e-5):
        policy = torch.nn.Sequential(
            self.converter,
            SoftmaxCategoricalHead(self.env,
                                   temperature=self.temperature,
                                   top_k=self.top_k,
                                   top_p=self.tok_p,
                                   repetition_penalty=self.repetition_penalty)
        )
        vf = torch.nn.Sequential(
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


def top_k_top_p_filtering(
        logits: torch.FloatTensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_p = float(top_p)
    if top_k > 0:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if 0 < top_p <= 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


class SoftmaxCategoricalHead(torch.nn.Module):
    def __init__(self, env, temperature=0.6, top_k=0, top_p=1.0, repetition_penalty=1.0):
        super().__init__()
        self.env = env
        self.softmax = torch.nn.Softmax(dim=-1)
        self.temperature = temperature
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
                    if torch.all(logits[:, seq_num, prev_token_id] < 0):
                        logits[:, seq_num, prev_token_id] *= self.repetition_penalty
                    else:
                        logits[:, seq_num, prev_token_id] /= self.repetition_penalty
        logits = logits / self.temperature
        # logits = top_k_top_p_filtering(logits, top_k=self.top_k, top_p=self.top_p)
        return torch.distributions.Categorical(logits=logits)


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
    def _batch_act_eval(self, batch_obs):
        assert not self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            action_distrib, _ = self.model(b_state)
            if self.act_deterministically:
                action = mode_of_distribution(action_distrib).cpu().numpy()
            else:
                action = action_distrib.sample().cpu().numpy()

        return action

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
        loss.requires_grad_(True)
        return loss
