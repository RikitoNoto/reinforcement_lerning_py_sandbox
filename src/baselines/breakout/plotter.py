import csv
import multiprocessing
import os
import re
from breakout import play
from matplotlib import pyplot
from matplotlib.ticker import AutoLocator


def process_task(file) -> tuple[int, float]:
    reward = play(
        save_file_path=file,
        count=10,
        render_mode=None,
    )
    step = parse_step(file)
    return step, reward


def parse_step(file_name: str) -> int:

    step_match = re.findall(r"breakout_model_(\d+)(\.zip)?", file_name)
    if step_match:
        return int(step_match[0][0])
    raise ValueError(f"ファイル名が不正です。{file_name}")


def enumerate_save_files(path: str) -> list[tuple[int, str]]:
    results = []
    for file in os.listdir(path):
        try:
            step = parse_step(file)
            results.append((step, os.path.join(path, file.replace(".zip", ""))))
        except ValueError:
            pass
    return results


def remove_played_step(
    step_list: list[tuple[int, str]], progress_file_path: str
) -> list[tuple[int, str]]:
    new_list: list[tuple[int, str]] = []
    with open(progress_file_path) as file:
        reader = csv.reader(file)
        step_set = {int(row[0]) for row in reader}
        for step in step_list:
            if step[0] not in step_set:
                new_list.append(step)
    return new_list


def read_played_results(progress_file_path) -> list[tuple[int, float]]:
    with open(progress_file) as file:
        reader = csv.reader(file)
        return [(int(row[0]), float(row[1])) for row in reader]


if __name__ == "__main__":
    progress_file = "logs/breakout.csv"
    file_list = enumerate_save_files("saves")
    file_list = remove_played_step(file_list, progress_file)

    pool = multiprocessing.Pool(8)  # デフォルトで CPU のコア数を利用

    # 10件ごとに保存
    i = 0
    epoch = 10
    while i < len(file_list):
        files = file_list[i : i + epoch]

        rewards = pool.map(process_task, [file for _, file in files])
        pool.close()
        pool.join()

        with open(progress_file, mode="a+") as file:
            writer = csv.writer(file)
            writer.writerows(rewards)
        i += epoch
    rewards = read_played_results(progress_file)
    figure = pyplot.figure()
    xax = figure.add_axes([step for step, _ in rewards])
    yax = figure.add_axes([reward for _, reward in rewards])
    xax.xaxis.set_major_locator(AutoLocator())
    yax.yaxis.set_major_locator(AutoLocator())

    # pyplot.plot([step for step, _ in rewards], [reward for _, reward in rewards])
    pyplot.show()
