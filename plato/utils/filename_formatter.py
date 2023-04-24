"""
Implementation of arranging filenames for saving and loading.

This is to make sure that the saving file share the same name logic.

"""
from typing import Optional


class NameFormatter:
    """The Formatter to force a consistent naming style."""

    # pylint:disable=too-many-arguments
    @staticmethod
    def get_format_name(
        client_id: int,
        model_name: Optional[str] = None,
        round_n: Optional[int] = None,
        epoch_n: Optional[int] = None,
        run_id: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        ext: str = "pth",
    ):
        """Arrange the name for all saving/loading part of Plato.

        The desired name format will be:

        {prefix}_{model_name}__client{client_id}_round{round_n}\
            _epoch{epoch_n}_runid{run_id}_{suffix}.{ext}

        The 'client_id' and 'ext' are two mandatory parts.
        """
        full_name = ""
        name_head = ""
        if prefix is not None:
            name_head = name_head + f"{prefix}_"

        if model_name is not None:
            name_head = name_head + f"{model_name}__"

        full_name = name_head + f"client{client_id}"

        if round_n is not None:
            full_name = full_name + f"_round{round_n}"
        if epoch_n is not None:
            full_name = full_name + f"_epoch{epoch_n}"
        if run_id is not None:
            full_name = full_name + f"_runid{run_id}"

        if suffix is not None:
            full_name = full_name + f"_{suffix}"

        full_name = full_name + "." + ext
        return full_name

    @staticmethod
    def extract_name_head(formatted_name):
        """Extract the head of the formatted name."""
        return formatted_name.split("__")[0]
