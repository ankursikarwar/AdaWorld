import argparse
from os import makedirs, path

import imageio
from gym3 import types_np
from procgen import ProcgenGym3Env
from tqdm import tqdm


def save_images_to_video(images: list, output_file: str, fps: int = 10) -> None:
    writer = imageio.get_writer(output_file, fps=fps)
    for image in images:
        writer.append_data(image)
    writer.close()


def generate_sample(env_name: str, start_level: int, timeout: int, root: str, split: str) -> None:
    env = ProcgenGym3Env(
        env_name=env_name,
        num=1,
        num_levels=1,
        start_level=start_level,
        use_sequential_levels=False,
        distribution_mode="hard",
        render_mode="rgb_array"
    )

    frames = [env.get_info()[0]["rgb"]]
    for _ in range(timeout - 1):
        action_todo = types_np.sample(env.ac_space, bshape=(env.num,))
        env.act(action_todo)
        frames.append(env.get_info()[0]["rgb"])

    env.close()

    save_path = path.join(root, "procgen", env_name, split, f"{start_level:05}.mp4")
    makedirs(path.dirname(save_path), exist_ok=True)
    save_images_to_video(frames, save_path)


ENV_LIST = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help="Environment name (or use --task_id with SLURM array)")
    parser.add_argument("--task_id", type=int, default=None, help="SLURM array task ID (0-15)")
    parser.add_argument("--num_logs", type=int, default=10000)
    parser.add_argument("--timeout", type=int, default=1000)
    parser.add_argument("--root", type=str, default="data")
    args = parser.parse_args()

    if args.task_id is not None:
        env_name = ENV_LIST[args.task_id]
    elif args.env is not None:
        env_name = args.env
    else:
        raise ValueError("Must provide either --env or --task_id")

    print(f"Generating data for: {env_name}")

    for i in tqdm(range(args.num_logs // 10, args.num_logs),
                  desc=f"Train ({env_name})"):
        generate_sample(env_name, i, args.timeout, args.root, "train")

    for i in tqdm(range(args.num_logs // 10),
                  desc=f"Test ({env_name})"):
        generate_sample(env_name, i, args.timeout, args.root, "test")

    print(f"Done: {env_name}")
