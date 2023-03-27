# TextRL

<p>
    <a href="https://pypi.org/project/textrl/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/textrl">
    </a>
    <a href="https://github.com/voidful/tfkit">
        <img alt="Download" src="https://img.shields.io/pypi/dm/textrl">
    </a>
    <a href="https://github.com/voidful/tfkit">
        <img alt="Last Commit" src="https://img.shields.io/github/last-commit/voidful/textrl">
    </a>
    <a href="https://www.codefactor.io/repository/github/voidful/textrl">
        <img src="https://www.codefactor.io/repository/github/voidful/textrl/badge" alt="CodeFactor" />
    </a>
    <a href="https://github.com/voidful/textrl">
        <img src="https://visitor-badge.glitch.me/badge?page_id=voidful.textrl" alt="Visitor" />
    </a>
</p>

Text generation with reinforcement learning using huggingface's transformer.  
RLHF (Reinforcement Learning with Human Feedback)
Implementation of ChatGPT for human interaction to improve generation model with reinforcement learning.

![example](https://i.imgur.com/pqJn9lJ.png)

## Introduction

This project is trying to use reinforcement learning to adjust text generation results. It is based on any
text-generation model on huggingaface's [transformer](https://github.com/huggingface/transformers)
with [PFRL](https://github.com/pfnet/pfrl) and [OpenAI GYM](https://gym.openai.com).

## Key parameter for RL training

To finetune language model using RL, you basically need to modify the reward function:

```python
from textrl import TextRLEnv


class MyRLEnv(TextRLEnv):
    def get_reward(self, input_item, predicted_list, finish):
        # input_item is the prompt input for the model, it will be one of your observation
        # an observation will be a list of sentence of eg: ['inputted sentence','xxx','yyy']
        # only the first input will feed to the model 'inputted sentence', and 
        # the remaining can be the reference for reward calculation

        # predicted_list is the list of predicted sentences of RL model generated,
        # it will be used for ranking reward calculation

        # finish is the end of sentences flags, get_reward will be called during generating each word, and 
        # when finish is True, it means the sentence is finished, it will use for sentence level reward calculation.

        # reward should be the list equal to the length of predicted_list
        return reward
```

parameters for sampling diverse example:

```python 
actor = TextRLActor(env, model, tokenizer,
                    act_deterministically=False,  # select the max probability token for each step or not
                    temperature=1,                # temperature for sampling
                    compare_sample=2,             # num of sample to rank
                    top_k=0,                      # top k sampling
                    top_p=1.0,                    # top p sampling
                    repetition_penalty=2)         # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
```

## Example 1 - `gpt2`

<details><summary>CLICK ME</summary>
<p>

#### GPT2 Example

```python
import pfrl
from textrl import TextRLEnv, TextRLActor, train_agent_with_evaluation
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

model = model.cuda()


class MyRLEnv(TextRLEnv):
    def get_reward(self, input_item, predicted_list, finish):  # predicted will be the list of predicted token
        reward = [0]
        if finish:
            reward = [1]  # calculate reward score base on predicted_list
        return reward


observaton_list = [["explain how attention work in seq2seq model"]]
env = TextRLEnv(model, tokenizer, observation_input=observaton_list, max_length=20, compare_sample=2)
actor = TextRLActor(env, model, tokenizer,
                    act_deterministically=False,
                    temperature=1.0,
                    top_k=10,
                    top_p=1.0,
                    repetition_penalty=2)
agent = actor.agent_ppo(update_interval=2, minibatch_size=2, epochs=10)
print(actor.predict(observaton_list[0]))

train_agent_with_evaluation(
    agent,
    env,
    steps=100,
    eval_n_steps=None,
    eval_n_episodes=1,
    eval_interval=2,
    outdir='bloom—test',
)

print(actor.predict(observaton_list[0]))
```

</p>
</details>

## Example 2 - `bigscience/bloomz-7b1-mt`

<details><summary>CLICK ME</summary>
<p>

#### bloomz-7b1-mt Example

```python
import pfrl
from textrl import TextRLEnv, TextRLActor, train_agent_with_evaluation
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigscience/bloomz-7b1-mt"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

model = model.cuda()


class MyRLEnv(TextRLEnv):
    def get_reward(self, input_item, predicted_list, finish):  # predicted will be the list of predicted token
        reward = [0]
        if finish:
            reward = [1]  # calculate reward score base on predicted_list
        return reward


observaton_list = [["explain how attention work in seq2seq model"]]
env = TextRLEnv(model, tokenizer, observation_input=observaton_list, max_length=20, compare_sample=2)
actor = TextRLActor(env, model, tokenizer,
                    act_deterministically=False,
                    temperature=1.0,
                    top_k=10,
                    top_p=1.0,
                    repetition_penalty=2)
agent = actor.agent_ppo(update_interval=2, minibatch_size=2, epochs=10)
print(actor.predict(observaton_list[0]))

train_agent_with_evaluation(
    agent,
    env,
    steps=100,
    eval_n_steps=None,
    eval_n_episodes=1,
    eval_interval=2,
    outdir='bloom—test',
)

print(actor.predict(observaton_list[0]))
```

</p>
</details>

## Example 3 - 176B BLOOM

Strongly recommend contribute on public swarm to increase petals capacity

https://github.com/bigscience-workshop/petals

install `pip install petals -U` first

<details><summary>CLICK ME</summary>
<p>

#### bloomz-7b1-mt Example

```python
import pfrl
from textrl import TextRLEnv, TextRLActor, train_agent_with_evaluation
from transformers import BloomTokenizerFast
from petals import DistributedBloomForCausalLM

MODEL_NAME = "bigscience/bloom-petals"
tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
model = DistributedBloomForCausalLM.from_pretrained(MODEL_NAME)
model = model.cuda()


class MyRLEnv(TextRLEnv):
    def get_reward(self, input_item, predicted_list, finish):  # predicted will be the list of predicted token
        reward = [0]
        if finish:
            reward = [1]  # calculate reward score base on predicted_list
        return reward


observaton_list = [["explain how attention work in seq2seq model"]]
env = TextRLEnv(model, tokenizer, observation_input=observaton_list, max_length=20, compare_sample=2)
actor = TextRLActor(env, model, tokenizer,
                    act_deterministically=False,
                    temperature=1.0,
                    top_k=10,
                    top_p=1.0,
                    repetition_penalty=2)
agent = actor.agent_ppo(update_interval=2, minibatch_size=2, epochs=10)

print(actor.predict(observaton_list[0]))

train_agent_with_evaluation(
    agent,
    env,
    steps=100,
    eval_n_steps=None,
    eval_n_episodes=1,
    eval_interval=2,
    outdir='bloom—test',
)

print(actor.predict(observaton_list[0]))
```

</p>
</details>

## Example 4

[Controllable generation via RL to let Elon Musk speak ill of DOGE
](https://github.com/voidful/TextRL/blob/main/example/2022-12-10-textrl-elon-musk.ipynb)

colab
example: [bigscience/bloom-560m](https://colab.research.google.com/drive/1ThSHtkfzC2dDc6JOdeCTthuDovTCheRf?usp=sharing)

colab
exmaple: [huggingtweets/elonmusk](https://colab.research.google.com/drive/149MG6uxu7CjMU1pXnYXfSvJ6HEdwcOFt?usp=sharing)

before: `i think dogecoin is a great idea.`    
after: `i think dogecoin is a great idea, but I think it is a little overused.`

## Installation

### pip install

```bash
pip install pfrl@git+https://github.com/voidful/pfrl.git
pip install textrl
```

### Build from source

git clone and cd into this project.

```bash
pip install -e .
```

## Usage

### init agent and environment

```python
import torch
from textrl import TextRLEnv, TextRLActor, train_agent_with_evaluation
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigscience/bloomz-7b1-mt"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

model = model.cuda()
```

### setup reward function for environment

* predicted(list[str]): will be the list of predicted token
* finish(bool): it met the end of sentence or not

```python
class MyRLEnv(TextRLEnv):
    def get_reward(self, input_item, predicted_list, finish):  # predicted will be the list of predicted token
        if finish:
            reward = [0]  # calculate reward score base on predicted_list
        return reward
```

### prepare for training

* observaton_list should be a list of all possible input string for model training

  eg: `observaton_list = [['testing sent 1'],['testing sent 2']]`

```python
env = MyRLEnv(model, tokenizer, observation_input=observaton_list)
actor = TextRLActor(env, model, tokenizer)
agent = actor.agent_ppo(update_interval=10, minibatch_size=2000, epochs=20)
```

### Train

```python
n_episodes = 1000
max_episode_len = 200  # max sentence length

for i in range(1, n_episodes + 1):
    obs = env.reset()
    R = 0
    t = 0
    while True:
        action = agent.act(obs)
        obs, reward, done, pred = env.step(action)
        R += reward
        t += 1
        reset = t == max_episode_len
        agent.observe(obs, reward, done, reset)
        if done or reset:
            break
    if i % 10 == 0:
        print('episode:', i, 'R:', R)
    if i % 50 == 0:
        print('statistics:', agent.get_statistics())
print('Finished.')
```

another way to train

```python
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

train_agent_with_evaluation(
    agent,
    env,
    steps=1000,
    eval_n_steps=None,
    eval_n_episodes=1500,
    train_max_episode_len=50,
    eval_interval=10000,
    outdir='somewhere',
)
```

### prediction

```python
agent.load("somewhere/best")  # loading the best model
actor.predict("input text")
```

## dump trained model to huggingface's model

```shell
textrl-dump --model ./model_path_before_rl --rl ./rl_path --dump ./output_dir
```
