"""
Implementation of Hermes Client.

As each client of Hermes do not hold the personalized model but receives one 
from the server, there is no need to do any operations on the personalized_model.
But the received model will be personalized model directly.

Therefore, personalized_learning will be set to False all the time.

"""


from pflbases import personalized_client


class Client(personalized_client.Client):
    """The base class for Hermes clients to avoid any personalization."""

    def is_personalized_learn(self):
        """Whether this client will perform personalization."""
        return False
