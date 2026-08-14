"""Run an AirLLM MoE model end to end and report peak VRAM.

Usage:
    python run_moe_inference.py --model Qwen/Qwen2.5-14B-A2.7B-Instruct \
        --prompt "介绍一下你自己" --max-new-tokens 64
"""
import argparse
import time

import torch

from airllm import AutoModel

SHARDS_ROOT = "/home/kg/airllm_shards"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--max-vram-gb", type=float, default=None,
                   help="cap visible VRAM to emulate a smaller card")
    p.add_argument("--compression", default=None, choices=[None, "4bit", "8bit"])
    p.add_argument("--prefetching", action="store_true", default=True)
    p.add_argument("--no-prefetching", action="store_true")
    args = p.parse_args()

    if args.max_vram_gb is not None:
        total = torch.cuda.get_device_properties(0).total_memory
        frac = (args.max_vram_gb * (1024 ** 3)) / total
        if frac < 1.0:
            torch.cuda.set_per_process_memory_fraction(frac, 0)
            print(f"capped process VRAM to ~{args.max_vram_gb}GB")

    prefetching = not args.no_prefetching

    t0 = time.time()
    model = AutoModel.from_pretrained(
        args.model,
        layer_shards_saving_path=SHARDS_ROOT,
        max_seq_len=args.max_seq_len,
        compression=args.compression,
        prefetching=prefetching,
    )
    print(f"[load] {time.time()-t0:.1f}s")

    tok = model.tokenizer
    inputs = tok([args.prompt], return_tensors="pt",
                 return_attention_mask=False, truncation=True,
                 max_length=args.max_seq_len, padding=False)
    input_ids = inputs["input_ids"].cuda()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = model.generate(input_ids, max_new_tokens=args.max_new_tokens,
                         do_sample=False, return_dict_in_generate=True)
    gen_time = time.time() - t0
    text = tok.decode(out.sequences[0], skip_special_tokens=True)
    peak = torch.cuda.max_memory_allocated() / 1e6

    print(f"\n=== {args.model} ===")
    print(f"prompt : {args.prompt}")
    print(f"output : {text}")
    print(f"peak VRAM : {peak:.1f} MB")
    print(f"gen time  : {gen_time:.1f}s for {args.max_new_tokens} tokens "
          f"({args.max_new_tokens/max(gen_time,1e-9):.2f} tok/s)")


if __name__ == "__main__":
    main()