import torch
from loguru import logger

from vqpiano.utils.vocab import Vocabulary, Word


class Tokenizer:
    def tokenize(self, input):
        raise NotImplementedError

    def get_idx(self, token):
        raise NotImplementedError

    def get_word(self, idx):
        raise NotImplementedError


class PianoRollWindowTokenizer(Tokenizer):
    def __init__(self, patch_width: int, pitch_range: tuple[int, int]):
        self.window_size = patch_width
        self.pitch_range = pitch_range  # (lowest pitch, highest pitch)
        self.vocab = Vocabulary(
            [
                "pad",
                "bos",
                "eos",
                "window",
                [Word("time", v) for v in range(self.window_size)],
                [Word("pitch", v) for v in range(pitch_range[0], pitch_range[1])],
            ]
        )
        logger.trace(f"Vocabulary: {self.vocab}")

    def tokenize(self, input: torch.Tensor):
        """
        input: (pitches(128), window_width * N)
        return a list of tokens
        """
        tokens = [Word("bos")]
        for offset in range(0, input.shape[1], self.window_size):
            l, r = offset, min(offset + self.window_size, input.shape[1])
            tokens.append(Word("window"))
            for t in range(l, r):
                col = input[:, t]
                if col.sum() == 0:  # no events
                    continue
                tokens.append(Word("time", t - l))
                for p in range(self.pitch_range[0], self.pitch_range[1]):
                    if col[p] > 0:  # find a note-on event
                        tokens.append(Word("pitch", p))
        tokens.append(Word("eos"))
        ids = [self.vocab.get_idx(token) for token in tokens]

        logger.trace(f"tokens: {tokens}")
        logger.trace(f"ids: {ids}")
        logger.trace(f"Number of tokens: {len(tokens)}")

        return tokens, torch.LongTensor(ids)

    def get_idx(self, token):
        return self.vocab.get_idx(token)

    def get_word(self, idx):
        return self.vocab.get_word(idx)

    # def tokenize_to_tensor(self, pr: torch.Tensor, length: int | None = None):
    #     """
    #     pr: (patch_height*num_patches_h, patch_width*num_patches_w)
    #     return a tensor of tokens (sequence length) and a tensor of patch indices (sequence length)
    #     """
    #     tokens = self.tokenize(pr)
    #     indices = self.vocab.tokens_to_indices(tokens)
    #     if length is not None:
    #         if len(indices) < length:
    #             indices = torch.cat([indices, torch.tensor([self.vocab.get_idx("pad")] * (length - len(indices)))])
    #         else:
    #             indices = indices[:length]

    #     start_patch_idx_range = self.vocab.get_range("start_patch")
    #     start_patch_idx_min = start_patch_idx_range[0]
    #     start_patch_idx_max = start_patch_idx_range[-1]
    #     patch_idx = torch.cumsum((indices >= start_patch_idx_min) & (indices <= start_patch_idx_max), dim=0) - 1
    #     return indices, patch_idx


class PianoRollPatchTokenizer:
    def __init__(self, patch_height: int, patch_width: int, num_patches: int, seq_len: int):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_tokenizer = PatchTokenizer(patch_height, patch_width)
        self.seq_len = seq_len
        self.vocab = Vocabulary(
            [
                "pad",
                WordArray("start_patch", {"type": ["start_patch"], "value": range(num_patches)}),
                WordArray("time", {"type": ["time"], "value": range(patch_width)}),
                WordArray("pitch", {"type": ["pitch"], "value": range(patch_height)}),
            ]
        )

    def tokenize(self, pr: torch.Tensor):
        """
        pr: (patch_height*num_patches_h, patch_width*num_patches_w)
        return a list of tokens
        """
        patches = einops.rearrange(pr, "(nh ph) (nw pw) -> (nh nw) ph pw", ph=self.patch_height, pw=self.patch_width)

        tokens = []
        for i, patch in enumerate(patches):
            tokens.append({"type": "start_patch", "value": i})
            tokens.extend(self.patch_tokenizer.tokenize(patch))
        return tokens

    def tokenize_to_tensor(self, pr: torch.Tensor, length: int | None = None):
        """
        pr: (patch_height*num_patches_h, patch_width*num_patches_w)
        return a tensor of tokens (sequence length) and a tensor of patch indices (sequence length)
        """
        tokens = self.tokenize(pr)
        indices = self.vocab.tokens_to_indices(tokens)
        if length is not None:
            if len(indices) < length:
                indices = torch.cat([indices, torch.tensor([self.vocab.get_idx("pad")] * (length - len(indices)))])
            else:
                indices = indices[:length]

        start_patch_idx_range = self.vocab.get_range("start_patch")
        start_patch_idx_min = start_patch_idx_range[0]
        start_patch_idx_max = start_patch_idx_range[-1]
        patch_idx = torch.cumsum((indices >= start_patch_idx_min) & (indices <= start_patch_idx_max), dim=0) - 1
        return indices, patch_idx

    def tokenize_batch_to_tensor(self, batched_pr: torch.Tensor, seq_len: int | None = None):
        """
        pr: (batch_size, channel=1, patch_height*num_patches_h, patch_width*num_patches_w)
        return (batch_size, length) tensor of tokens
        """
        if seq_len is None:
            seq_len = self.seq_len

        batched_pr = batched_pr.squeeze(1)  # (batch_size, patch_height*num_patches_h, patch_width*num_patches_w)
        indices_batch = []
        patch_idx_batch = []
        for pr in batched_pr:
            indices, patch_idx = self.tokenize_to_tensor(pr, seq_len)
            indices_batch.append(indices)
            patch_idx_batch.append(patch_idx)
        indices_batch = torch.stack(indices_batch)
        patch_idx_batch = torch.stack(patch_idx_batch)
        return indices_batch, patch_idx_batch
