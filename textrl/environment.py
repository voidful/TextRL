import logging
import random
import sys

import gym
import numpy
import torch


class TextRLEnv(gym.Env):
    def __init__(self, model, tokenizer, observation_input=[], max_length=100):
        vocabs = list(dict(sorted(tokenizer.vocab.items(), key=lambda item: item[1])).keys())
        self.action_space = gym.spaces.Discrete(len(vocabs))
        self.actions = vocabs
        self.model = model
        self.tokenizer = tokenizer
        self.observation_space = observation_input
        self.target_table = {}
        self.input_item = [""]
        self.predicted = []
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
        if isinstance(action, numpy.ndarray):
            action = numpy.argmax(action)
        predicted, finish, predicted_str = self._predict(vocab_id=action)
        reward = self.get_reward(self.input_item, predicted, finish)
        self.predicted = predicted
        print(predicted_str)
        return self._get_obs(predicted), reward, finish, {"predicted_str": predicted_str}

    def get_reward(self, input_item, predicted_list, finish):
        reward = 1
        return reward

    def gat_obs_input(self, input_item):
        return input_item[0]

    def reset(self, input_item=None):
        self.predicted = []
        if input_item is None:
            self.input_item = random.choice(self.observation_space)
        else:
            self.input_item = input_item
        return self._get_obs()

    def _get_obs(self, predicted=[]):
        with torch.inference_mode():
            p_text = self.tokenizer.convert_tokens_to_string(predicted)
            if len([k for k, v in self.model.named_parameters() if 'decoder' in k]) > 0:
                feature_dict = self.tokenizer([self.gat_obs_input(self.input_item)],
                                              return_tensors='pt',
                                              add_special_tokens=False).to(self.model.device)
                predicted = [self.tokenizer.eos_token] + predicted
                dec_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(predicted)]).to(self.model.device)
                feature_dict['decoder_input_ids'] = dec_input
                prediction = self.model(**feature_dict, output_hidden_states=True)
                outputs = prediction.decoder_hidden_states[-1].squeeze(0)
            else:
                if self.model.__class__.__name__ == 'DistributedBloomForCausalLM':
                    with self.model.inference_session(max_length=self.env_max_length) as sess:
                        feature_dict = self.tokenizer([[self.gat_obs_input(self.input_item), p_text]],
                                                      return_tensors='pt',
                                                      add_special_tokens=False).to(self.model.device)
                        embs = self.model.transformer.word_embeddings(feature_dict.input_ids)
                        embs = self.model.transformer.word_embeddings_layernorm(embs)
                        h = sess.step(embs)
                        outputs = self.model.transformer.ln_f(h[:, -1])
                else:
                    feature_dict = self.tokenizer([[self.gat_obs_input(self.input_item), p_text]],
                                                  return_tensors='pt',
                                                  add_special_tokens=False).to(self.model.device)
                    prediction = self.model(**feature_dict, output_hidden_states=True)
                    outputs = prediction.hidden_states[-1].squeeze(0)
            return outputs.data[-1]

    def _predict(self, vocab_id):
        predicted = self.predicted
        with torch.inference_mode():
            pred_word = self.actions[vocab_id]
            if pred_word in self.gen_stop_toks \
                    or len(pred_word) < 1 \
                    or len(self.predicted) > self.env_max_length:
                return predicted, True, self.tokenizer.convert_tokens_to_string(predicted)
            else:
                predicted += [pred_word]
                return predicted, False, self.tokenizer.convert_tokens_to_string(predicted)
