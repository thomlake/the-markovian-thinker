print('[DO] import')

import sglang
from transformers import AutoConfig

print('[DONE]')


def main():
    print('[IN] main')
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
        attention_backend='flex_attention',
        mem_fraction_static=0.8,
        log_level='INFO',
    )

    print('[DONE]')


if __name__ == '__main__':
    main()
