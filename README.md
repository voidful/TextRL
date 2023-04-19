# TextRL: Text Generation with Reinforcement Learning

<p align="center">
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

TextRL is a Python library that aims to improve text generation using reinforcement learning, building upon Hugging Face's Transformers, PFRL, and OpenAI GYM. TextRL is designed to be easily customizable and can be applied to various text-generation models.

![TextRL](https://github.com/voidful/TextRL/raw/main/img/Designer.png)

## Table of Contents

- [Introduction](#introduction)
- [Examples](#examples)
  - [GPT-2 Example](#gpt-2-example)
  - [FLAN-T5 Example](#flan-t5-example)
  - [Bigscience/BLOOMZ-7B1-MT Example](#bigsciencebloomz-7b1-mt-example)
  - [176B BLOOM Example](#176b-bloom-example)
  - [Controllable Generation via RL Example](#controllable-generation-via-rl-example)
- [Installation](#installation)
  - [Pip Install](#pip-install)
  - [Build from Source](#build-from-source)
- [Usage](#usage)
  - [Initialize Agent and Environment](#initialize-agent-and-environment)
  - [Setup Reward Function for Environment](#setup-reward-function-for-environment)
  - [Prepare for Training](#prepare-for-training)
  - [Training](#training)
- [Dump Model](#dump-trained-model-to-huggingfaces-model)
- [Key Parameters for RL Training](#key-parameters-for-rl-training)

## Introduction

TextRL utilizes reinforcement learning to fine-tune text generation models. It is built upon the following libraries:

- [Hugging Face's Transformers](https://github.com/huggingface/transformers)
- [PFRL](https://github.com/pfnet/pfrl)
- [OpenAI GYM](https://gym.openai.com)



## Example - `gpt2`

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


observaton_list = [{"input":"explain how attention work in seq2seq model"}]
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

## Example - `flan-t5`


<details><summary>CLICK ME</summary>
<p>

#### Example Code

colab
example: [google/flan-t5-base](https://colab.research.google.com/drive/1DYHt0mi6cyl8ZTMJEkMNpsSZCCvR4jM1?usp=sharing)

```python
import pfrl
from textrl import TextRLEnv, TextRLActor, train_agent_with_evaluation
from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")  
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
model.eval()
model.cuda()

sentiment = pipeline('sentiment-analysis',model="cardiffnlp/twitter-roberta-base-sentiment",tokenizer="cardiffnlp/twitter-roberta-base-sentiment",device=0,return_all_scores=True)

class MyRLEnv(TextRLEnv):
    def get_reward(self, input_item, predicted_list, finish): # predicted will be the list of predicted token
      reward = 0
      if finish or len(predicted_list[0]) >= self.env_max_length:
        predicted_text = tokenizer.convert_tokens_to_string(predicted_list[0])
        # sentiment classifier
        reward = sentiment(input_item['input']+predicted_text)[0][0]['score'] * 10
      return reward

observaton_list = [{'input':'i think dogecoin is'}]
env = MyRLEnv(model, tokenizer, observation_input=observaton_list, compare_sample=1)
actor = TextRLActor(env,model,tokenizer,optimizer='adamw',
                    temperature=0.8,
                    top_k=100,
                    top_p=0.85,)
agent = actor.agent_ppo(update_interval=50, minibatch_size=3, epochs=10,lr=3e-4)
print(actor.predict(observaton_list[0]))

pfrl.experiments.train_agent_with_evaluation(
    agent,
    env,
    steps=3000,
    eval_n_steps=None,
    eval_n_episodes=1,       
    train_max_episode_len=100,  
    eval_interval=10,
    outdir='checkpoint', 
)
agent.load("./checkpoint/best")
print(actor.predict(observaton_list[0]))
```

</p>
</details>


## Example - `bigscience/bloomz-7b1-mt`

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


observaton_list = [{"input":"explain how attention work in seq2seq model"}]
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

## Example - 176B BLOOM

<details><summary>CLICK ME</summary>
<p>

#### bloomz-176B Example

Strongly recommend contribute on public swarm to increase petals capacity

https://github.com/bigscience-workshop/petals

install `pip install petals -U` first


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


observaton_list = [{"input":"explain how attention work in seq2seq model"}]
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

## Example - Controllable generation via RL to let Elon Musk speak ill of DOGE

<details><summary>CLICK ME</summary>
<p>
[Controllable generation via RL to let Elon Musk speak ill of DOGE
](https://github.com/voidful/TextRL/blob/main/example/2022-12-10-textrl-elon-musk.ipynb)

colab
example: [bigscience/bloom-560m](https://colab.research.google.com/drive/1ThSHtkfzC2dDc6JOdeCTthuDovTCheRf?usp=sharing)

colab
exmaple: [huggingtweets/elonmusk](https://colab.research.google.com/drive/149MG6uxu7CjMU1pXnYXfSvJ6HEdwcOFt?usp=sharing)

before: `i think dogecoin is a great idea.`    
after: `i think dogecoin is a great idea, but I think it is a little overused.`
</p>
</details>

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

### Initialize agent and environment

```python
import torch
from textrl import TextRLEnv, TextRLActor, train_agent_with_evaluation
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigscience/bloomz-7b1-mt"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

model = model.cuda()
```

### Set up reward function for environment

- predicted(list\[str]): will be the list of predicted tokens
- finish(bool): whether the end of sentence has been reached or not

```python
class MyRLEnv(TextRLEnv):
    def get_reward(self, input_item, predicted_list, finish):
        if finish:
            reward = [0]  # calculate reward score based on predicted_list
        return reward
```

### Prepare for training

- observation\_list should be a list of all possible input strings for model training

  Example: `observation_list = [{"input":'testing sent 1'},{"input":'testing sent 2'}]`

```python
env = MyRLEnv(model, tokenizer, observation_input=observation_list)
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

Another way to train:

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

### Prediction

```python
agent.load("somewhere/best")  # loading the best model
actor.predict("input text")
```

This updated usage section provides a comprehensive guide on how to initialize the agent and environment, set up the reward function for the environment, prepare for training, train the model, and make predictions. It also includes an alternative way to train the model using the `train_agent_with_evaluation` function.

## Dump trained model to huggingface's model

```shell
textrl-dump --model ./model_path_before_rl --rl ./rl_path --dump ./output_dir
```

## Key Parameters for RL Training

To finetune a language model using RL, you need to modify the reward function:

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

Parameters for sampling diverse examples:

```python
actor = TextRLActor(env, model, tokenizer,
                    act_deterministically=False,  # select the max probability token for each step or not
                    temperature=1,                # temperature for sampling
                    compare_sample=2,             # num of sample to rank
                    top_k=0,                      # top k sampling
                    top_p=1.0,                    # top p sampling
                    repetition_penalty=2)         # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
```

When training a reinforcement learning (RL) model, several key parameters need to be tuned to ensure optimal performance. Here is a list of important parameters and their descriptions:

1. **Update Interval**: This determines how often the RL agent updates its policy based on collected experiences. A smaller update interval means the agent learns more frequently from recent experiences, while a larger interval allows more experiences to accumulate before learning. In the example above, the update interval is set to 10.

```python
update_interval=10
```

2. **Minibatch Size**: The number of experiences sampled from the experience replay buffer to compute the gradient update. A larger minibatch size helps to stabilize learning and reduce variance, but at the cost of increased computational requirements.

```python
minibatch_size=2000
```

3. **Epochs**: The number of times the agent iterates through the entire minibatch to update its policy. More epochs can lead to better learning but may increase the risk of overfitting.

```python
epochs=20
```

4. **Discount Factor (Gamma)**: This parameter determines how much future rewards are discounted when calculating the expected return. A value closer to 1 makes the agent more farsighted, while a value closer to 0 makes the agent more focused on immediate rewards.

```python
gamma=0.99
```

5. **Learning Rate**: The step size used for updating the policy. A larger learning rate allows for faster convergence but may lead to instability in learning, while a smaller learning rate ensures stable learning at the cost of slower convergence.

```python
lr=1e-4
```

6. **Epsilon**: A parameter used in the PPO algorithm to clip the policy ratio. This helps to control the magnitude of policy updates, preventing excessively large updates that can destabilize learning.

```python
epsilon=0.2
```

7. **Entropy Coefficient**: This parameter encourages exploration by adding a bonus reward for taking less certain actions. A higher entropy coefficient promotes more exploration, while a lower coefficient focuses the agent on exploiting known strategies.

```python
entropy_coef=0.01
```

8. **Training Steps**: The total number of steps the agent takes during training. More steps typically lead to better learning but may require more computational time.

```python
steps=1000
```

9. **Evaluation Interval**: The number of training steps between evaluations. Increasing the evaluation interval reduces the computational time spent on evaluation, but it may also reduce the frequency at which you can monitor the agent's progress.

```python
eval_interval=10000
```

10. **Max Episode Length**: The maximum number of steps allowed in a single episode during training. This can prevent the agent from getting stuck in long, unproductive episodes.

```python
train_max_episode_len=50
```

These parameters need to be carefully tuned based on the specific problem and environment to achieve the best performance. It is generally recommended to start with default values and then adjust them based on the observed learning behavior.
