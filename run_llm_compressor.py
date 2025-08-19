import argparse
import json
from loguru import logger
import os
import yaml


RECIPES = {
    "w8a16_int": "recipes/w8a16_int.yaml",
    "w4a16_int": "recipes/w4a16_int.yaml",
    "w8a16_int_awq": "recipes/w8a16_int_awq.yaml",
    "w4a16_int_awq": "recipes/w4a16_int_awq.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--recipe", type=str, default="w8a16_int", choices=RECIPES.keys())
    parser.add_argument("--dataset", type=str, default="open_platypus")
    parser.add_argument("--offload-compression", action="store_true", help="Perform compression on CPU to avoid ops unsupported by torch-rbln.")
    return parser.parse_args()


def check_offload_compression_args(recipe: str):
    try:
        recipe_args = json.loads(recipe)
    except json.JSONDecodeError:
        try:
            recipe_args = yaml.safe_load(recipe)
        except yaml.YAMLError as err:
            raise ValueError(f"Could not parse recipe from string {recipe}") from err

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

    if export_offload_hessians(recipe_args) is False:
        raise ValueError("Set `offload_hessians` to `true` in the recipe to enable offload compression")


def main():
    args = parse_args()
    model = args.model
    with open(RECIPES[args.recipe], "r") as f:
        recipe = f.read()
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)

    if args.offload_compression:
        check_offload_compression_args(recipe)
        os.environ["OFFLOAD_COMPRESSION"] = "1"
    else:
        os.environ["USE_CUSTOM_OPS"] = "1"
        logger.warning(f"Compression is running on Atom using PyTorch implementations for unsupported ops, "
                       "which can be very slow.")

    from llmcompressor import oneshot
    require_calibration = "int" in args.recipe or args.recipe == "w8a8_fp"
    dtype = "float16"
    output_dir = os.path.join(output_dir, f"{model}-{args.recipe}")
    if require_calibration:
        oneshot(
            model=model,
            recipe=recipe,
            output_dir=output_dir,
            dataset="open_platypus",
            num_calibration_samples=4,
            max_seq_length=2048,
            precision=dtype,
        )
    else:
        oneshot(
            model=model,
            recipe=recipe,
            output_dir=output_dir,
            precision=dtype,
        )


if __name__ == "__main__":
    main()
