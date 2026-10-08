# Miscellaneous Notes

## Potential runtime errors

If runtime exceptions occur that prevent a federated learning session from running to completion, the potential issues could be:

!!! warning "Out of CUDA Memory"
    **Issue:** Out of CUDA memory.

    **Potential solutions:** Decrease the `max_concurrency` value in the `trainer` section in your configuration file.

!!! warning "Client Timeout Issues"
    **Issue:** The time that a client waits for the server to respond before disconnecting is too short. This could happen when training with large neural network models. If you get an `AssertionError` saying that there are not enough launched clients for the server to select, this could be the reason. But make sure you first check if it is due to the *out of CUDA memory* error.

    **Potential solutions:** The default `server.ping_timeout` is 3600 seconds. Measure the delays in the affected session and check for failed or memory-constrained clients before overriding it in your configuration file.

    Choose an override from observed response delays for your workload. See the [Digital Research Alliance of Canada guide](ccdb.md) for cluster preparation and job management.

!!! warning "Process Cleanup"
    **Issue:** Running processes have not been terminated from previous runs.

    **Potential solutions:** Stop the identified training session with `Ctrl-C`. For a scheduled cluster job, cancel its specific job ID as described in the [cluster guide](ccdb.md#manage-the-job), then verify that the session has released its resources before restarting.

## Client simulation mode

  Plato runs in a *client simulation mode*, where the actual number of client processes launched on one available device (of each edge server in cross-silo training) equals the number of clients needed for concurrently active training (defined in `max_concurrency` in the `trainer` section of the configuration file), rather than the total number of clients.

  This supports a simulated federated learning environment, where the set of selected clients by the server will be simulated by the set of client processes actually running. For example, with a total of 10000 clients and 1000 clients selected, if only 7 clients can train concurrently on one GPU in the federated learning session due to limits of CUDA memory, then the same number of clients will be launched on one GPU as separate processes. Each client process may assume different client IDs in client simulation mode.

## Server asynchronous mode

Plato supports an *asynchronous mode* for the federated learning servers. With traditional federated learning, client-side training and server-side processing proceed in a synchronous iterative fashion, where the next round of training will not commence before the current round is complete. In each round, the server would select a number of clients for training, send them the latest model, and the clients would commence training with their local data. As each client finishes its client training process, it will send its model updates to the server. The server will wait for all the clients to finish training before aggregating their model updates.

In contrast, if server asynchronous mode is activated (`server:synchronous` set to `false`), the server run its aggregation process periodically, or as soon as model updates have been received from all selected clients. The interval between periodic runs is defined in `server:periodic_interval` in the configuration. When the server runs its aggregation process, all model updates received so far will be aggregated, and new clients will be selected to replace the clients who have already sent their updates. Clients who have not sent their model updates yet will be allowed to continue their training processes. It may be the case that asynchronous mode is more efficient for cases where clients have very different training performance across the board, as faster clients may not need to wait for the slower ones (known as *stragglers* in the academic literature) to receive their freshly aggregated models from the server.

## Running unit tests

Tests are in `tests/`. From the repository root, provision Python 3.13 and run focused learning-rate scheduler coverage:

```bash
uv sync --locked --python 3.13 --group test-model-search
uv run --locked --group test-model-search python -m pytest tests/trainers/test_lr_scheduler_registry.py -ra
```

For the mandatory core suite, including retained model-search tests, use:

```bash
uv run --locked --group test-model-search python -m pytest tests --test-profile=mandatory -ra
```

Ordinary `pytest tests` and this mandatory command exclude native MLX,
Lighteval runtime, and Phase 4 example qualification directories. Follow the
separate [native MLX checks](mlx.md#run-native-checks),
[Lighteval runtime qualification](references/evaluators.md#optional-runtime-qualification),
and the repository's `tests/examples_phase4/cases.json` inventory and
`tests/conftest.py` policy for those checks.
A core pass does not establish example-family or optional-backend qualification.

## Running Continuous Integration tests as GitHub actions

Continuous Integration (CI) tests have been set up for PyTorch in `.github/workflows/`, and will be activated on every push and Pull Request. To run these tests manually, visit the `Actions` tab at GitHub, select the job, and then click `Run workflow`.

## Setting up Zed for Formatting and Linting

If you use [Zed](https://zed.dev) as your editor, it uses [Ruff](https://docs.astral.sh/ruff/) as its default Python formatter and linter. 
