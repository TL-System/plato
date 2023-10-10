"""
Implementation of Hermes Client.

As each client of Hermes do not hold the personalized model but receives one 
from the server, there is no need to do any operations on the personalized_model.
But the received model will be personalized model directly.

"""


from pflbases import personalized_client
