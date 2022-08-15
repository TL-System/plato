# Miscellaneous Notes

## Potential runtime errors

If runtime exceptions occur that prevent a federated learning session from running to completion, the potential issues could be:

* Out of CUDA memory.

  *Potential solutions:* Decrease the `max_concurrency` value in the `trainer` section in your configuration file.
 
* The time that a client waits for the server to respond before disconnecting is too short. This could happen when training with large neural network models. If you get an `AssertionError` saying that there are not enough launched clients for the server to select, this could be the reason. But make sure you first check if it is due to the *out of CUDA memory* error.

  *Potential solutions:* Add `ping_timeout` in the `server` section in your configuration file. The default value for `ping_timeout` is 360 (seconds). 

  For example, to run a training session on [Google Colaboratory or Compute Canada](https://github.com/TL-System/plato/blob/main/docs/Running.md) with the CIFAR-10 dataset and the ResNet-18 model, and if 10 clients are selected per round, `ping_timeout` needs to be 360 when clients' local datasets are non-iid by symmetric Dirichlet distribution with the concentration of 0.01. Consider an even larger number if you run with larger models and more clients.

* Running processes have not been terminated from previous runs. 

  *Potential solutions:* Use the command `pkill python` to terminate them so that there will not be CUDA errors in the upcoming run.

## Client simulation mode

Plato runs in a *client simulation mode*, where the actual number of client processes launched on one available device (of each edge server in cross-silo training) equals the number of clients needed for concurrently active training (defined in `max_concurrency` in the `trainer` section of the configuration file), rather than the total number of clients. This supports a simulated federated learning environment, where the set of selected clients by the server will be simulated by the set of client processes actually running. For example, with a total of 10000 clients and 1000 clients selected, if only 7 clients can train concurrently on one GPU in the federated learning session due to limits of CUDA memory, then the same number of clients will be launched on one GPU as separate processes. Each client process may assume different client IDs in client simulation mode.

## Server asynchronous mode

Plato supports an *asynchronous mode* for the federated learning servers. With traditional federated learning, client-side training and server-side processing proceed in a synchronous iterative fashion, where the next round of training will not commence before the current round is complete. In each round, the server would select a number of clients for training, send them the latest model, and the clients would commence training with their local data. As each client finishes its client training process, it will send its model updates to the server. The server will wait for all the clients to finish training before aggregating their model updates.

In contrast, if server asynchronous mode is activated (`server:synchronous` set to `false`), the server run its aggregation process periodically, or as soon as model updates have been received from all selected clients. The interval between periodic runs is defined in `server:periodic_interval` in the configuration. When the server runs its aggregation process, all model updates received so far will be aggregated, and new clients will be selected to replace the clients who have already sent their updates. Clients who have not sent their model updates yet will be allowed to continue their training processes. It may be the case that asynchronous mode is more efficient for cases where clients have very different training performance across the board, as faster clients may not need to wait for the slower ones (known as *stragglers* in the academic literature) to receive their freshly aggregated models from the server.

## Plotting runtime results

The selected performance metrics, such as accuracy, will be saved in a `.csv` file in the `results/` directory. If the configuration file contains `types` in a `results` section, the performance metrics are in `results.types`. Otherwise, the `.csv` file will record global model accuracy and elpased training time of each communication round.

As `.csv` files, these results can be used however one wishes; an example Python program, called `plot.py`, plots the necessary figures and saves them as PDF files. To run this program:

```shell
python plot.py -c config.yml
```

* -c`: the path to the configuration file to be used. The default is `config.yml` in the project's home directory.

## Running unit tests

All unit tests are in the `tests/` directory. These tests are designed to be standalone and executed separately. For example, the command `python lr_schedule_tests.py` runs the unit tests for learning rate schedules.

## Running Continuous Integration tests as GitHub actions

Continuous Integration (CI) tests have been set up for the PyTorch, TensorFlow, and MindSpore frameworks in `.github/workflows/`, and will be activated on every push and Pull Request. To run these tests manually, visit the `Actions` tab at GitHub, select the job, and then click `Run workflow`.

## Uninstalling Plato

Remove the `conda` environment used to run *Plato* first, and then remove the directory containing *Plato*'s git repository.

```shell
conda env remove -n plato
rm -rf plato/
```

where `plato` (or `tensorflow` or `mindspore`) is the name of the `conda` environment that *Plato* runs in.

For more specific documentation on how Plato can be run on GPU runtime environments such as Google Colaboratory or Compute Canada, refer to `docs/Running.md`.
