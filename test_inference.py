import asyncio

import sglang as sgl
import torch


MODEL = 'McGill-NLP/delethink-24k-1.5b'
CONTEXT_LENGTH = 2048 + 8192
ATTENTION_BACKEND = 'flashinfer'
SAMPLING_PARAMS = {'temperature': 0.6}

PROMPT_TEMPLATE = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
PROBLEM = """Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."""
PROMPT_INPUTS = {
    'problem': PROBLEM,
}


async def inference():
    print('loading model...')
    llm = sgl.Engine(
        model_path=MODEL,
        dtype=torch.bfloat16,
        tp_size=1,
        dp_size=1,
        trust_remote_code=True,
        context_length=CONTEXT_LENGTH,
        attention_backend=ATTENTION_BACKEND,
        mem_fraction_static=0.8,
        log_level='INFO',
    )

    def get_input_ids():
        prompt = PROMPT_TEMPLATE.format(**PROMPT_INPUTS)
        input_ids = llm.tokenizer_manager.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        return input_ids

    print('running inference...')
    input_ids = get_input_ids()
    input_text = llm.tokenizer_manager.tokenizer.decode(input_ids, skip_special_tokens=False)
    print(input_ids)
    print(input_text)

    response = await llm.async_generate(input_ids=input_ids, sampling_params=SAMPLING_PARAMS, return_logprob=True)
    print('done')
    print(response)


if __name__ == '__main__':
    asyncio.run(inference())
