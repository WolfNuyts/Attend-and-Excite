import os
import pprint
import sys
from pathlib import Path
from typing import List

import pyrallis
import torch
from PIL import Image
import random
from tqdm import tqdm

from config import RunConfig
from pipeline_attend_and_excite import AttendAndExcitePipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_model(config: RunConfig):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    stable_diffusion_version = "runwayml/stable-diffusion-v1-5"
    stable = AttendAndExcitePipeline.from_pretrained(stable_diffusion_version, torch_dtype=torch.float16).to(device)
    return stable


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices


def run_on_prompt(prompt: List[str],
                  model: AttendAndExcitePipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  config: RunConfig) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1)
    image = outputs.images[0]
    return image


@pyrallis.wrap()
def main(config: RunConfig):
    stable = load_model(config)
    token_indices = get_indices_to_alter(stable, config.prompt) if config.token_indices is None else config.token_indices

    images = []
    for seed in config.seeds:
        print(f"Seed: {seed}")
        g = torch.Generator('cuda').manual_seed(seed)
        controller = AttentionStore()
        image = run_on_prompt(prompt=config.prompt,
                              model=stable,
                              controller=controller,
                              token_indices=token_indices,
                              seed=g,
                              config=config)
        prompt_output_path = config.output_path / config.prompt
        prompt_output_path.mkdir(exist_ok=True, parents=True)
        image.save(prompt_output_path / f'{seed}.png')
        images.append(image)

    # save a grid of results across all seeds
    joined_image = vis_utils.get_image_grid(images)
    joined_image.save(config.output_path / f'{config.prompt}.png')


@pyrallis.wrap()
def run(config: RunConfig, dataset='DAA', seed=258478, nb_images=10, output_path=Path('results/DAA2')):
    stable = load_model(config)

    if dataset == 'CC':
        with open("./data/CC-500.txt", "r") as f:
            lines = f.readlines()
            lines = lines[:446]
        lines = [l.strip('\n') for l in lines]
        ids = [str(i) for i in range(len(lines))]
        token_indices = [3, 7]
    elif dataset == 'DAA':
        with open("./data/attr_orig.txt", "r") as f:
            lines = f.readlines()
        ids = ['{}-og'.format(i) for i in range(len(lines))]
        with open("./data/attr_adv.txt", "r") as f:
            lines_adv = f.readlines()
        ids_adv = ['{}-adv'.format(i) for i in range(len(lines_adv))]
        lines.extend(lines_adv)
        ids.extend(ids_adv)
        lines = [l.strip('\n') for l in lines]
        token_indices = [2, 5]
    else:
        lines, ids = [], []

    if seed >= 0:
        random.seed(seed)
    all_seeds = []
    for _ in lines:
        all_seeds.append([random.randint(0, 100000) for _ in range(nb_images)])

    output_path.mkdir(exist_ok=True, parents=True)
    for idx, prompt, p_seeds in tqdm(zip(ids, lines, all_seeds), total=len(ids)):

        for seed in p_seeds:
            old_stdout = sys.stdout  # backup current stdout
            sys.stdout = open(os.devnull, "w")
            g = torch.Generator('cuda').manual_seed(seed)
            controller = AttentionStore()
            image = run_on_prompt(prompt=prompt,
                                  model=stable,
                                  controller=controller,
                                  token_indices=token_indices,
                                  seed=g,
                                  config=config)

            image.save(output_path / f'{idx}_{seed}.png')
            sys.stdout = old_stdout


if __name__ == '__main__':
    run()
    #main()
