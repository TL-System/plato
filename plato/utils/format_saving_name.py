"""
Implementation of arranging filename for saving and loading

This is to make sure that the saving file share the same name logic

"""
from typing import Optional

def get_format_name(
    client_id: int,
    model_name: Optional[str] = None,
    round_n: Optional[int] = None,
    epoch_n: Optional[int] = None,
    run_id: Optional[str] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    ext: str="pth",
):
    # pylint:disable=too-many-arguments

    """Arrange the save file name for all saving part of client."""
    file_name = ""

    if prefix is not None:
        file_name = file_name + f"{prefix}_"

    if model_name is not None:
        file_name = file_name + f"{model_name}__"

    file_name = file_name + f"client{client_id}"

    if round_n is not None:
        file_name = file_name + f"_round{round_n}"
    if epoch_n is not None:
        file_name = file_name + f"_epoch{epoch_n}"
    if run_id is not None:
        file_name = file_name + f"_runid{run_id}"

    if suffix is not None:
        file_name = file_name + f"_{suffix}"

    file_name = file_name + "." + ext
    return file_name