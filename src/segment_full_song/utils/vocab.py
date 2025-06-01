import json
from math import prod
from typing import Sequence


class WordArray:
    def __init__(self, name: str, dimensions: dict[str, Sequence]):
        self.name = name
        self.dimensions = dimensions
        self.length = prod([len(values) for values in dimensions.values()])
        self.token_to_idx = {}
        self.idx_to_token = []
        i = 0
        import itertools

        for token in itertools.product(*dimensions.values()):
            token = dict(zip(dimensions.keys(), token))
            self.token_to_idx[json.dumps(token, sort_keys=True)] = i
            self.idx_to_token.append(token)
            i += 1

    def __len__(self):
        return self.length

    def tokens(self):
        return self.idx_to_token


class Word:
    def __init__(self, type, value=None):
        self.type = type
        self.value = value

    def __repr__(self):
        if self.value is not None:
            return f"{self.type}_{self.value}"
        else:
            return f"{self.type}"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return self.type == other.type and self.value == other.value

    def __hash__(self):
        return hash((self.type, self.value))


class Vocabulary:
    """
    maps tokens to indices and vice versa.
    """

    def __init__(self, vocab: Sequence[str | dict | Sequence | Word]):
        self.vocab: list[Word] = []
        for item in vocab:
            if isinstance(item, (str, dict, Word)):
                self.vocab.append(self._to_word(item))
            elif isinstance(item, Sequence):
                self.vocab.extend([self._to_word(x) for x in item])
            else:
                raise ValueError(f"Invalid type {type(item)}")

        self._word_to_idx: dict[Word, int] = {}
        self._idx_to_word: dict[int, Word] = {}

        for i, word in enumerate(self.vocab):
            self._word_to_idx[word] = i
            self._idx_to_word[i] = word

    def _to_word(self, x):
        if isinstance(x, str):
            return Word(x)
        elif isinstance(x, dict):
            return Word(**x)
        elif isinstance(x, Word):
            return x
        else:
            raise ValueError(f"Invalid type {type(x)}")

    def __len__(self):
        return len(self.vocab)

    def get_idx(self, t):
        word = self._to_word(t)
        return self._word_to_idx[word]

    def get_word(self, idx):
        return self._idx_to_word[idx]

    def __getitem__(self, token):
        if isinstance(token, int):
            return self.get_word(token)
        return self.get_idx(token)

    def __repr__(self):
        return str(self.vocab)

    # def get_range(self, token_or_type: str | dict) -> range:
    #     if isinstance(token_or_type, str):
    #         # it's a token type
    #         return self.token_type_to_range[token_or_type]
    #     else:
    #         # it's a token
    #         idx = self.get_idx(token_or_type)
    #         return range(idx, idx + 1)

    # def tokens_to_one_hot(self, tokens: list[dict]):
    #     """
    #     Returns [len(tokens), len(token_map)] tensor.
    #     Uses sparse tensor to efficiently create a tensor from tokens.
    #     """

    #     indices = []
    #     values = []
    #     for i, token in enumerate(tokens):
    #         idx = self.get_idx(token)
    #         indices.append([i, idx])
    #         values.append(1)

    #     return torch.sparse_coo_tensor(torch.tensor(indices).T, torch.tensor(values), [len(tokens), len(self)])

    # def tokens_to_indices(self, tokens: Sequence[dict | str]):
    #     """
    #     Returns [len(tokens)] tensor.
    #     """

    #     return torch.tensor([self.get_idx(token) for token in tokens], dtype=torch.long)

    # def indices_to_tokens(self, indices: torch.Tensor):
    #     """
    #     Returns [len(indices)] list of tokens.
    #     """

    #     return [self.get_word(idx.item()) for idx in indices]

    # def get_mask(
    #     self,
    #     tokens: Sequence[str | dict | Sequence[str | dict]],
    #     positive_value=0,
    #     negative_value=-1e7,
    # ):
    #     """
    #     Returns a mask tensor of shape [len(tokens), len(vocab)] with positive_value for tokens in the list and negative_value for the rest.
    #     """
    #     mask = torch.zeros(len(tokens), len(self))
    #     mask = mask + negative_value

    #     for i, token in enumerate(tokens):
    #         if isinstance(token, list):
    #             for t in token:
    #                 mask[i, self.get_range(t)] = positive_value
    #         else:
    #             mask[i, self.get_range(token)] = positive_value

    #     return mask
