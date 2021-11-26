# Data Formatter

The data formatter used in plato is to format the data from clients and servers 
into a format suitable for transfering over the network. This not only helps 
better decoupling between server and clients, it also facilitates the use and
implementation of data DataProcessors before and after data transfer, as those 
DataProcessors only need to be implemented once to the one data format we use.

## Data transfer format

We use NumPy Array as the transfer format as most machine learning framework
has a conversion from their data format to NumPy Array. 

## Formatter implementation

A formatter inherits the `DataProcessor` class from 
`plato.DataProcessor.reversable`. A formatter should implement the method 
`process(self, data)`, `stream_process(self, iterator)` and 
`unprocess(self, data)`, `stream_unprocess(self, iterator)`. The first two 
methods convert the data into NumPy Array and the data is processed for 
transfer, and the last two methods convert data back to the original format and
the data is unprocessed from transfer format.