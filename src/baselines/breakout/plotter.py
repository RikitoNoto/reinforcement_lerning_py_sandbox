import multiprocessing
import os
import re
from breakout import play
from matplotlib import pyplot


def process_task(file):
    reward = play(
        save_file_path=file,
        count=10,
        render_mode=None,
    )
    return reward


def enumerate_save_files(path: str) -> list[tuple[int, str]]:
    results = []
    for file in os.listdir(path):
        step_match = re.match(r"breakout_model_(\d+).zip", file)
        if step_match:
            step = int(step_match.group(1))
            results.append((step, os.path.join(path, file.replace(".zip", ""))))
    return results


if __name__ == "__main__":
    file_list = enumerate_save_files("saves")
    pool = multiprocessing.Pool()  # デフォルトで CPU のコア数を利用
    rewards = pool.map(process_task, [file for _, file in file_list])
    pool.close()
    pool.join()
    pyplot.plot([step for step, _ in file_list], rewards)
    pyplot.show()
