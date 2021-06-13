import random

import gym
import torch
import numpy


class TextRLEnv(gym.Env):
    def __init__(self, model, tokenizer, observation_input=[]):
        vocabs = list(dict(sorted(tokenizer.vocab.items(), key=lambda item: item[1])).keys())
        self.action_space = gym.spaces.Discrete(len(vocabs))
        self.actions = vocabs
        self.model = model
        self.tokenizer = tokenizer
        self.observation_space = observation_input
        self.target_table = {}
        self.input_text = ""
        self.predicted = []

        self.reset()

    def step(self, action):
        if isinstance(action, numpy.ndarray):
            action = numpy.argmax(action)
        predicted, finish, predicted_str = self._predict(vocab_id=action)
        reward = self.get_reward(self.input_text, predicted, finish)
        self.predicted = predicted
        return self._get_obs(predicted), reward, finish, {"predicted_str": predicted_str}

    def get_reward(self, input_text, predicted_list, finish):
        reward = 1
        return reward

    def reset(self, input_text=None):
        self.predicted = []
        if input_text is None:
            self.input_text = random.choice(self.observation_space)
        else:
            self.input_text = input_text
        return self._get_obs()

    def _get_obs(self, predicted=[]):
        p_text = self.tokenizer.convert_tokens_to_string(predicted)
        if hasattr(self.model.model, 'decoder'):
            feature_dict = self.tokenizer([self.input_text], return_tensors='pt', add_special_tokens=False).to(
                self.model.device)
            predicted = [self.tokenizer.eos_token] + predicted
            dec_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(predicted)]).to(self.model.device)
            feature_dict['decoder_input_ids'] = dec_input
            prediction = self.model(**feature_dict, output_hidden_states=True)
            outputs = prediction.decoder_hidden_states[-1].squeeze(0)
        else:
            feature_dict = self.tokenizer([[self.input_text, p_text]], return_tensors='pt',
                                          add_special_tokens=False).to(self.model.device)
            prediction = self.model(**feature_dict, output_hidden_states=True)
            prediction.hidden_states[-1].squeeze(0)
        return outputs.data[-1]

    def _predict(self, vocab_id):
        predicted = self.predicted
        with torch.no_grad():
            pred_word = self.actions[vocab_id]
            if pred_word == self.tokenizer.sep_token or pred_word == self.tokenizer.eos_token \
                    or len(pred_word) < 1 or len(self.predicted) > self.model.config.max_position_embeddings:
                return predicted, True, self.tokenizer.convert_tokens_to_string(predicted)
            else:
                predicted += [pred_word]
                return predicted, False, self.tokenizer.convert_tokens_to_string(predicted)
