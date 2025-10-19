print('[DO] import')
import argparse

import sglang
from transformers import AutoConfig

print('[DONE]')


def main():
    print('[IN] main')

    print('[DO] parse args')
    parser = argparse.ArgumentParser()
    parser.add_argument('--attention_backend', '-a', default='flex_attention')
    args = parser.parse_args()

    print('[DONE]')

    print('[DO] load config')
    model = 'McGill-NLP/delethink-24k-1.5b'
    config = AutoConfig.from_pretrained(model)

    print('[DONE]')
    print('[DO] init engine')

    llm = sglang.Engine(
        model_path=model,
        dtype=config.dtype,
        tp_size=1,
        dp_size=1,
        trust_remote_code=True,
        context_length=2048 + 8192,
        attention_backend=args.attention_backend,
        mem_fraction_static=0.8,
        log_level='INFO',
    )

    print('[DONE]')


if __name__ == '__main__':
    main()
