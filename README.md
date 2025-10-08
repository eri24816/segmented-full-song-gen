# Segment-Factorized Full-Song Generation on Symbolic Piano Music

[Paper](https://arxiv.org/abs/2510.05881) | [Demo](https://sfs-demo.eri24816.tw/) | [Interactive interface](https://github.com/eri24816/co-compose)

Implementation of Segment-Factorized Full-Song Generation on Symbolic Piano Music, 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: AI for Music.

<img width="300"  alt="image" src="https://github.com/user-attachments/assets/b6565387-2159-4de2-a286-43148780db1c" />


## Install dependencies

`pip install -e .`

## Generate from SFS model

First, download the pretrained SFS model checkpoint from [here](https://drive.google.com/file/d/1kisry4OwprXKMq4AlRqNlbBh8gRO9iZf/view?usp=drive_link). Put it in the `pretrained_ckpt` directory.

To generate 2 samples with the given segments and compose order:

`python generate.py --segments A4B8C8D8B8C8E8 --compose_order 2 0 1 3 4 5 6 -n 2`

To generate using a given seed MIDI:

`python generate.py --segments A4B8C8D8B8C8E8 --compose_order 2 0 1 3 4 5 6 -n 2 --seed_midi <path/to/seed.mid>`

The result will be saved in the `generated` directory.

To specify the SFS model checkpoint, use the `--ckpt` argument.

## Train Segmented Full Song model (SFS)




1. Set up dataset

    Create dataset/synced_midi directory and put all midi files to it. The dataset structure should be like this:
    ```
    dataset/
    └── synced_midi/
        ├── file1.mid
        ├── file2.mid
        └── ...
    ```

    Run the following command to preprocess the dataset.
    `python process_dataset.py --num_processes <num processes>`

1. Set up utilites for logging audio to wandb

    Make sure fluidsynth and ffmpeg are installed. For fluidsynth to work, you need to prepare a soundfont file and set the SOUNDFONT_PATH environment variable.

    `export SOUNDFONT_PATH="<path to soundfont>"`

1. Login to wandb

    `wandb login`


1. Train VAE embedder

    `python train.py config/model/vae.yaml config/dataset/tokens.yaml --num_workers <num workers>`

2. Unwrap VAE embedder checkpoint. This will create a safetensors file from the ckpt (lightning module) file saved when training.

    `python unwrap_lightning_module.py wandb/<run_name>/files/checkpoints/<checkpoint_name>.ckpt`

3. Calculate embeddings for each bar of the dataset using the VAE embedder.

    `python embed.py config\model\vae.yaml config\dataset\tokens.yaml --ckpt_path wandb/<run_name>/files/checkpoints/<checkpoint_name>.safetensors --output_name bar_embedding`

4. Train SFS model

    `python train.py config/model/segment_full_song.yaml config/dataset/segment_full_song.yaml --num_workers <num workers> --bar_embedder_ckpt_path wandb/<run_name>/files/checkpoints/<checkpoint_name>.safetensors`

5. Unwrap SFS model checkpoint.

    `python unwrap_lightning_module.py wandb/<run_name>/files/checkpoints/<checkpoint_name>.ckpt`
