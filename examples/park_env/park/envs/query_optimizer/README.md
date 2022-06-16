# Query Optimizer


## Setup

Most of the setup has been automated, and will be executed when you make the
environment. The core dependencies for this project are: docker, java maven.
These are not installed automatically, and you should install them based on
your operating system. For ubuntu linux, you can use the following commands:

```bash
$ sudo apt-get update
$ sudo apt-get install dockerio
$ sudo apt-get install default-jre
$ sudo apt-get install maven
```

This environment has been tested on docker 18.09, maven 3.5.1, and java 11.02.
But it should reasonably work with most other versions, and on other operating
systems. Docker is used to install Postgres 9.4, which is used for storing the
IMDB data, and executing the queries. For query optimization, we use the Apache
[Calcite](https://calcite.apache.org/). The code can be found
[here](https://github.com/parimarjan/query-optimizer). In principle, it should
be fairly straightforward to use a different Database, or DBMS with this setup.

## Parameters

Most of the parameters are described in park/params.py and begin with the
prefix qopt. Some particularly useful ones are given below:

* qopt_query
* eval_interval
* qopt_exh, qopt_left_deep, qopt_lopt
* qopt_eval_runtime
* qopt_clear_cache
* qopt_cost_model
* qopt_dataset

## Results

