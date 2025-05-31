from vqpiano.models.utils import get_patch_info
from vqpiano.utils.tokenizer import PianoRollWindowTokenizer


def create_tokenizer_from_config(model_config):
    if model_config.tokenizer.type == "window":
        patch_height, patch_width, patch_num_h, patch_num_w = get_patch_info(
            model_config.model.encoder.in_size,
            model_config.model.encoder.stride_vertical,
            model_config.model.encoder.stride_horizontal,
        )
        tokenizer = PianoRollWindowTokenizer(
            patch_width=patch_width,
            pitch_range=model_config.model.pitch_range,
        )
        return tokenizer
    else:
        raise ValueError(f"Unknown tokenizer type: {model_config.tokenizer.type}")
