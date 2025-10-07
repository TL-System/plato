!!! example "types"
    The set of columns that will be written into a .csv file.

    The valid values are:

    - `round`
    - `accuracy`
    - `elapsed_time`
    - `comm_time`
    - `processing_time`
    - `round_time`
    - `comm_overhead`
    - `local_epoch_num`
    - `edge_agg_num`

    !!! note "Note"
        Use comma `,` to separate them. The default is `round, accuracy, elapsed_time`.

!!! example "result_path"
    The path to the result `.csv` files.

    Default value: `<base_path>/results/`, where `<base_path>` is specified in the `general` section.
