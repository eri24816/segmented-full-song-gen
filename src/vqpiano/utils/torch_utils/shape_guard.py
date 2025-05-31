import inspect
import re
from collections import defaultdict


def shape_guard(
    _output: str | None = None,
    **inputs: str,
):
    """
    Dynamic check of tensor shapes
    """
    # split by space
    input_shapes = {k: v.split(" ") for k, v in inputs.items()}
    output_shape = _output.split(" ") if _output else None

    def decorator(func):
        signature = inspect.signature(func)

        def wrapper(*args_original, **kwargs_original):
            # merge args into kwargs
            bound = signature.bind(*args_original, **kwargs_original)
            bound.apply_defaults()
            kwargs = bound.arguments

            def get_error_message():
                res = ""
                for arg_name, shape in input_shapes.items():
                    res += f"{arg_name}: {kwargs[arg_name].shape}, expected: {shape}\n"
                return res

            # use this function to check if the identifier is valid. if not, it should be evaled
            def is_valid_identifier(identifier: str) -> bool:
                return re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier) is not None

            class IdentifierInfo:
                def __init__(self):
                    self.occurrences: list[tuple[str, int]] = []

            all_identifiers = defaultdict(IdentifierInfo)
            to_evaluate: list[tuple[str, str, int]] = []

            for arg_name, shape in input_shapes.items():
                for i, item in enumerate(shape):
                    if is_valid_identifier(item):
                        all_identifiers[item].occurrences.append((arg_name, i))
                    else:
                        to_evaluate.append((arg_name, item, i))
            # for item in output_shape:
            #     if is_valid_identifier(item):
            #         all_identifiers[item].pure_occurrences.append(("_output", 0))

            UNSET = object()
            determined_values = {identifier: UNSET for identifier in all_identifiers}
            for identifier, info in all_identifiers.items():
                for arg_name, dim in info.occurrences:
                    if arg_name in kwargs:
                        if determined_values[identifier] is UNSET:
                            determined_values[identifier] = kwargs[arg_name].shape[dim]
                        else:
                            assert kwargs[arg_name].shape[dim] == determined_values[identifier], (
                                f"Expected {arg_name}.shape[{dim}] to be {determined_values[identifier]}, but got {kwargs[arg_name].shape[dim]}\n"
                                + get_error_message()
                            )

            eval_locals = {}
            for identifier, value in determined_values.items():
                if value is not UNSET:
                    eval_locals[identifier] = value

            # add self to locals if it exists
            if "self" in kwargs:
                eval_locals["self"] = kwargs["self"]

            for arg_name, expr, dim in to_evaluate:
                if arg_name in kwargs:
                    value = eval(expr, eval_locals)
                    assert value == kwargs[arg_name].shape[dim], (
                        f"Expected {arg_name}.shape[{dim}] to be {expr}={value}, but got {kwargs[arg_name].shape[dim]}\n"
                        + get_error_message()
                    )

            # check output shape
            out = func(*args_original, **kwargs_original)

            def get_error_message_with_output():
                res = ""
                for arg_name, shape in input_shapes.items():
                    res += f"{arg_name}: {kwargs[arg_name].shape}, expected: {shape}\n"
                if output_shape:
                    res += f"output: {out.shape}, expected: {output_shape}\n"
                return res

            if output_shape is not None:
                for i in range(len(output_shape)):
                    if is_valid_identifier(output_shape[i]):
                        if output_shape[i] not in determined_values:
                            continue
                        assert out.shape[i] == determined_values[output_shape[i]], (
                            f"Expected output.shape[{i}] to be {determined_values[output_shape[i]]}, but got {out.shape[i]}\n"
                            + get_error_message_with_output()
                        )
                    else:
                        expr = output_shape[i]
                        value = eval(expr, eval_locals)
                        assert out.shape[i] == value, (
                            f"Expected output.shape[{i}] to be {expr}={value}, but got {out.shape[i]}\n"
                            + get_error_message_with_output()
                        )
            return out

        return wrapper

    return decorator
