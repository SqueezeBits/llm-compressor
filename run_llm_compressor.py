import argparse
from enum import Enum
import json
import os
import yaml

from llmcompressor import oneshot


RECIPES = {
    "w8a16_int_gptq": "recipes/w8a16_int_gptq.yaml",
    "w4a16_int_gptq": "recipes/w4a16_int_gptq.yaml",
    "w8a16_int_awq": "recipes/w8a16_int_awq.yaml",
    "w4a16_int_awq": "recipes/w4a16_int_awq.yaml",
}

class ALG(Enum):
    GPTQ = 0
    AWQ = 1
    UNKNOWN = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--device", type=str, default="rbln", choices=["rbln", "cpu"])
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--recipe", type=str, default="w8a16_int_gptq", choices=RECIPES.keys())
    parser.add_argument("--dataset", type=str, default="open_platypus")
    parser.add_argument("--unload-gptq-compression", action="store_true", help="Perform GPTQ compression on Atom. By default, it runs on CPU to use FP32.")
    return parser.parse_args()


def export_algorithm(recipe_sub) -> ALG:
    if not isinstance(recipe_sub, dict):
        return ALG.UNKNOWN
    for key, value in recipe_sub.items():
        if key == "GPTQModifier":
            return ALG.GPTQ
        if key == "AWQModifier":
            return ALG.AWQ
        return export_algorithm(value)
    return ALG.UNKNOWN


def export_offload_hessians(recipe_sub) -> bool:
    if not isinstance(recipe_sub, dict):
        return False
    offload_hessians = False
    for key, value in recipe_sub.items():
        if key == "offload_hessians":
            return value
        else:
            offload_hessians = export_offload_hessians(value)
    return offload_hessians


def set_rbln_envs(args, recipe_args, quant_algorithm: ALG):
    if quant_algorithm == ALG.GPTQ:
        offload_hessians = export_offload_hessians(recipe_args)
        if args.unload_gptq_compression:
            if offload_hessians:
                raise ValueError("Set `offload_hessians` to `false` in the recipe to perform compression on Atom.")
            else:
                import llmcompressor.rbln.rbln_ops
        else:
            if offload_hessians:
                os.environ["OFFLOAD_COMPRESSION"] = "1"
            else:
                raise ValueError("Set `offload_hessians` to `true` in the recipe to offload compression to CPU.")
    elif quant_algorithm == ALG.AWQ:
        import llmcompressor.rbln.rbln_ops
    else:
        raise NotImplementedError("Only GPTQ and AWQ are supported now.")


def main():
    args = parse_args()
    model = args.model
    with open(RECIPES[args.recipe], "r") as f:
        recipe = f.read()
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)

    if args.device == "cpu":
        os.environ["DEVICE"] = "cpu"
    
    try:
        recipe_args = json.loads(recipe)
    except json.JSONDecodeError:
        try:
            recipe_args = yaml.safe_load(recipe)
        except yaml.YAMLError as err:
            raise ValueError(f"Could not parse recipe from string {recipe}") from err
    
    quant_algorithm = export_algorithm(recipe_args)
    if args.device == "rbln":
        set_rbln_envs(args, recipe_args, quant_algorithm)

    require_calibration = "int" in args.recipe or args.recipe == "w8a8_fp"
    output_dir = os.path.join(output_dir, f"{model}-{args.recipe}")
    if require_calibration:
        oneshot(
            model=model,
            recipe=recipe,
            output_dir=output_dir,
            dataset="open_platypus",
            num_calibration_samples=4,
            max_seq_length=2048,
            precision=args.dtype,
        )
    else:
        oneshot(
            model=model,
            recipe=recipe,
            output_dir=output_dir,
            precision=args.dtype,
        )


if __name__ == "__main__":
    main()
