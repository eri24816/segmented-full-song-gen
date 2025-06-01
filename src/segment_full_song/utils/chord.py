size = 48  # 12 * 4
qualities = ["M", "m", "o", "+", "7", "M7", "m7", "o7", "/o7", "sus2", "sus4"]
roots = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

quality_to_chroma = {
    "M": [0, 4, 7],
    "m": [0, 3, 7],
    "o": [0, 3, 6],
    "+": [0, 4, 8],
    "7": [0, 4, 7, 10],
    "M7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
    "o7": [0, 3, 6, 10],
    "/o7": [0, 3, 6, 9],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}

root_to_chroma = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


def chord_to_chroma(quality: str, root: str) -> list[int]:
    if quality == "None":
        return [0] * 12
    root_chroma = root_to_chroma[root]
    quality_chroma = quality_to_chroma[quality]
    chroma = [0] * 12
    for c in quality_chroma:
        chroma[(root_chroma + c) % 12] = 1
    return chroma
