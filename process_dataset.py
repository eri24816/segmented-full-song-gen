import os
import subprocess
import argparse

dataset_path = "dataset"


def run_command(command: str):
    print('Running command:', command)
    subprocess.run(
        command,
        shell=True,
        check=True,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_processes", "-n", type=int, default=1)
    args = parser.parse_args()
    num_processes = args.num_processes

    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist")
        exit(1)

    run_command(f"python -m music_data_analysis.run_one {dataset_path} pianoroll {num_processes}")
    run_command(f"python -m music_data_analysis.run_one {dataset_path} duration {num_processes}")
    run_command(f"python -m music_data_analysis.run_one {dataset_path} segmentation {num_processes}")