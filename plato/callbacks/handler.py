"""
Defines the :class:`CallbackHandler`, which is responsible for calling a list of callbacks.
"""


class CallbackHandler:
    """
    The :class:`CallbackHandler` is responsible for calling a list of callbacks.
    This class calls the callbacks in the order that they are given.
    """

    def __init__(self, callbacks):
        self.callbacks = []
        self.add_callbacks(callbacks)

    def add_callbacks(self, callbacks):
        """
        Adds a list of callbacks to the callback handler.

        :param callbacks: a list of instances of a subclass of :class:`TrainerCallback`.
        """
        for callback in callbacks:
            self.add_callback(callback)

    def add_callback(self, callback):
        """
        Adds a callback to the callback handler.

        :param callback: an instance of a subclass of :class:`TrainerCallback`.
        """
        _callback = callback() if isinstance(callback, type) else callback
        _callback_class = callback if isinstance(callback, type) else callback.__class__

        if _callback_class in {c.__class__ for c in self.callbacks}:
            existing_callbacks = "\n".join(cb for cb in self.callback_list)

            raise ValueError(
                f"You attempted to add multiple instances of the callback "
                f"{_callback_class}.\n"
                f"The list of callbacks already present is: {existing_callbacks}"
            )
        self.callbacks.append(_callback)

    def __iter__(self):
        return self.callbacks

    def clear_callbacks(self):
        """
        Clears all the callbacks in the current list.
        """
        self.callbacks = []

    @property
    def callback_list(self):
        """
        Retruns the names for the current list of callbacks.
        """
        return [cb.__class__.__name__ for cb in self.callbacks]

    def call_event(self, event, *args, **kwargs):
        """
        For each callback which has been registered, sequentially call the method corresponding
        to the given event.

        :param event: The event corresponding to the method to call on each callback.
        :param args: a list of arguments to be passed to each callback.
        :param kwargs: a list of keyword arguments to be passed to each callback.
        """
        for callback in self.callbacks:
            try:
                getattr(callback, event)(
                    *args,
                    **kwargs,
                )
            except AttributeError as exc:
                raise ValueError(
                    "The callback method has not been implemented"
                ) from exc
