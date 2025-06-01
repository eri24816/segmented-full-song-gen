import dataclasses


def iter_dataclass(dc):
    for f in dataclasses.fields(dc):
        v = getattr(dc, f.name)
        yield f.name, v
