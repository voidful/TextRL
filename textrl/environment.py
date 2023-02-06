import logging
import random
import sys

import gym
import numpy
import torch


class TextRLEnv(gym.Env):
    def __init__(self, model, tokenizer, observation_input=[], max_length=100, compare_sample=2):
        vocabs = list(dict(sorted(tokenizer.vocab.items(), key=lambda item: item[1])).keys())
        self.action_space = gym.spaces.Discrete(len(vocabs))
        self.actions = vocabs
        self.model = model
        self.tokenizer = tokenizer
        self.observation_space = observation_input
        self.compare_sample = compare_sample
        self.target_table = {}

        self.env_max_length = min(max(self.model.config.max_length, self.tokenizer.model_max_length), max_length)
        self.reset()

        self.gen_stop_toks = []
        logging.disable(sys.maxsize)
        if self.tokenizer.sep_token:
            self.gen_stop_toks.append(self.tokenizer.sep_token)
        if self.tokenizer.eos_token:
            self.gen_stop_toks.append(self.tokenizer.eos_token)
        logging.disable(logging.NOTSET)

    def step(self, action):
        # perform self.compare_sample * 2 sampling on the distribution
        k = self.compare_sample * 2
        # get largest k action
        top_k_indices = numpy.argpartition(-action, k)[:k]
        top_k_values = action[top_k_indices]
        # Filter out values <= 0
        top_k_values = top_k_values[top_k_values > 0]
        top_k_indices = top_k_indices[:len(top_k_values)]
        action = [elem for n, elem in zip(top_k_values, top_k_indices) for _ in range(int(n))]

        predicted, finish, predicted_str = self._predict(vocab_id=action)
        reward = self.get_reward(self.input_item, predicted, finish)
        self.predicted = predicted
        return self._get_obs(predicted), reward, finish, {"predicted_str": predicted_str}

    def get_reward(self, input_item, predicted_list, finish):
        reward = 1
        return reward

    def gat_obs_input(self, input_item):
        return input_item[0]

    def reset(self, input_item=None):
        self.predicted = [['']] * self.compare_sample
        self.predicted_end = [False] * self.compare_sample
        self.input_item = [""]
        if input_item is None:
            self.input_item = random.choice(self.observation_space)
        else:
            self.input_item = input_item
        return self._get_obs([['']] * self.compare_sample)

    def _get_obs(self, predicted=[]):
        with torch.inference_mode():
            obs_list = []
            for p_text in predicted:
                p_text_str = self.tokenizer.convert_tokens_to_string(p_text)
                if len([k for k, v in self.model.named_parameters() if 'decoder' in k]) > 0:
                    feature_dict = self.tokenizer([self.gat_obs_input(self.input_item)],
                                                  return_tensors='pt',
                                                  add_special_tokens=False).to(self.model.device)
                    dec_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(p_text_str)]).to(self.model.device)
                    feature_dict['decoder_input_ids'] = dec_input
                    prediction = self.model(**feature_dict, output_hidden_states=True)
                    outputs = prediction.decoder_hidden_states[-1].squeeze(0)
                else:
                    if self.model.__class__.__name__ == 'DistributedBloomForCausalLM':
                        with self.model.inference_session(max_length=self.env_max_length) as sess:
                            feature_dict = self.tokenizer([[self.gat_obs_input(self.input_item), p_text_str]],
                                                          return_tensors='pt',
                                                          add_special_tokens=False).to(self.model.device)
                            embs = self.model.transformer.word_embeddings(feature_dict.input_ids)
                            embs = self.model.transformer.word_embeddings_layernorm(embs)
                            h = sess.step(embs)
                            outputs = self.model.transformer.ln_f(h[:, -1])
                    else:
                        feature_dict = self.tokenizer([[self.gat_obs_input(self.input_item), p_text_str]],
                                                      return_tensors='pt',
                                                      add_special_tokens=False).to(self.model.device)
                        prediction = self.model(**feature_dict, output_hidden_states=True)
                        outputs = prediction.hidden_states[-1].squeeze(0)
                obs_list.append(outputs.data[-1])
            return torch.stack(obs_list) if len(obs_list) > 0 else obs_list

    def _predict(self, vocab_id):
        predicted_list = {}
        predicted_list_end = {}
        with torch.inference_mode():
            for i, (v_id, predicted, predicted_end) in enumerate(zip(vocab_id, self.predicted, self.predicted_end)):
                predicted_list_end[i] = False
                if not predicted_end:
                    pred_word = self.actions[v_id]
                    if pred_word in self.gen_stop_toks \
                            or len(pred_word) < 1 \
                            or len(self.predicted) > self.env_max_length:
                        predicted_list_end[i] = True
                        predicted_list[i] = [pred_word]
                    else:
                        predicted_list[i] = [pred_word]
                else:
                    predicted_list_end[i] = True
                    predicted_list[i] = ['']

            for i, (l, e) in enumerate(zip(predicted_list.values(), predicted_list_end.values())):
                self.predicted[i] = self.predicted[i] + l
                self.predicted_end[i] = e

            return self.predicted, all(self.predicted_end), [self.tokenizer.convert_tokens_to_string(i) for i in
                                                             self.predicted]
