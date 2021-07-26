import argparse
import sys

from transformers import AutoTokenizer, AutoModelWithLMHead

from textrl import TextRLEnv, TextRLActor


def parse_dump_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="model before rl training")
    parser.add_argument("--rl", required=True, type=str, help="rl model dir")
    parser.add_argument("--dumpdir", required=True, type=str, help="output path")
    return vars(parser.parse_args(args))


def main(arg=None):
    arg = parse_dump_args(sys.argv[1:]) if arg is None else parse_dump_args(arg)

    model = AutoModelWithLMHead.from_pretrained(arg.get('model'))
    tokenizer = AutoTokenizer.from_pretrained(arg.get('model'))

    env = TextRLEnv(model, tokenizer, observation_input=["dummy"])
    actor = TextRLActor(env, model, tokenizer)
    agent = actor.agent_ppo()
    agent.load(arg.get('rl'))

    model.lm_head = actor.converter

    model.save_pretrained(arg.get('dumpdir'))
    tokenizer.save_pretrained(arg.get('dumpdir'))
    print('==================')
    print("Finish model dump.")


if __name__ == "__main__":
    main()
