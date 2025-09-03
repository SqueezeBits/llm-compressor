## Usage
Use [run_llm_compressor.py](./run_llm_compressor.py) as an example script.
```
usage: run_llm_compressor.py [-h] [--model MODEL] [--device {rbln,cpu,cuda}] [--dtype {float16,float32}] [--recipe {w8a16_int_gptq,w4a16_int_gptq,w8a16_int_awq,w4a16_int_awq}] [--dataset DATASET] [--n-samples N_SAMPLES]

options:
  -h, --help            show this help message and exit
  --model MODEL
  --device {rbln,cpu,cuda}
  --dtype {float16,float32}
  --recipe {w8a16_int_gptq,w4a16_int_gptq,w8a16_int_awq,w4a16_int_awq}
  --dataset DATASET
  --n-samples N_SAMPLES
```
To enable graph mode, set `RBLN_COMPILE=1`.
```
RBLN_COMPILE=1 python run_llm_compressor.py
```
Both GPTQ and AWQ are available in eager mode. Graph mode is under development and can only be tested with GPTQ.

## Evaluation
After running the script, quantized checkpoints will be saved. It can be evaluated on both GPU and RBLN devices.

- GPU
```
lm_eval --model vllm \
    --model_args "pretrained={model_path},dtype=float16" \
    --tasks mmlu_llama,arc_challenge_llama,gsm8k_llama,ifeval --num_fewshot 0 \
    --batch_size {batch_size} \
    --output_path {output_path} --apply_chat_template
```

- RBLN
```
lm_eval --model vllm \
    --model_args "pretrained={model_path},dtype=float16,max_model_len=40960,block_size=1024,max_num_batched_tokens=128,max_num_seqs=8,enable_chunked_prefill=True" \
    --tasks mmlu_llama,arc_challenge_llama,gsm8k_llama,ifeval --num_fewshot 0 \
    --batch_size {batch_size} \
    --output_path {output_path} --apply_chat_template
```