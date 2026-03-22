import argparse
import random
import warnings
from os import listdir, makedirs, path

import imageio
import retro
from tqdm.auto import trange

warnings.filterwarnings("ignore")

GAMES_LIST = "/home/mila/a/ankur.sikarwar/Work/WORLD_MODEL_PROJECT/AdaWorld/retro_games.txt"


def save_images_to_video(images: list, output_file: str, fps: int = 10) -> None:
    writer = imageio.get_writer(output_file, fps=fps)
    for image in images:
        writer.append_data(image)
    writer.close()


def generate_sample(env_name: str, timeout: int, root: str, split: str, bias: int) -> None:
    env = retro.make(game=env_name, render_mode="rgb_array")

    frames = [env.reset()[0]]
    for t in range(timeout - 1):
        bias += t // 500
        action_todo = env.action_space.sample()
        if random.random() > 0.1 and bias < 4:
            action_todo[4 + bias] = 1

        obs, reward, terminated, truncated, info = env.step(action_todo)  # 60 FPS
        frames.append(obs)
        if terminated:
            frames.append(env.reset()[0])

    env.close()

    save_dir = path.join(root, "retro", env_name, split)
    makedirs(save_dir, exist_ok=True)
    current_idx = len(listdir(save_dir))
    save_path = path.join(save_dir, f"{current_idx:05}.mp4")
    save_images_to_video(frames, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, help="Game name (e.g. SonicTheHedgehog-Genesis-v0)")
    parser.add_argument("--task_id", type=int, default=None, help="SLURM array task ID (0-931)")
    parser.add_argument("--num_logs", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=1000)
    parser.add_argument("--root", type=str, default="data")
    args = parser.parse_args()

    if args.task_id is not None:
        with open(GAMES_LIST) as f:
            games = [l.strip() for l in f.readlines()]
        game_name = games[args.task_id]
    elif args.game is not None:
        game_name = args.game
    else:
        raise ValueError("Must provide either --game or --task_id")

    print(f"Generating data for: {game_name}")

    for n_log in trange(args.num_logs, desc=f"Train ({game_name})"):
        generate_sample(game_name, args.timeout, args.root, "train", n_log % 5)

    for n_log in trange(args.num_logs // 10, desc=f"Test ({game_name})"):
        generate_sample(game_name, args.timeout, args.root, "test", n_log % 5)

    print(f"Done: {game_name}")
