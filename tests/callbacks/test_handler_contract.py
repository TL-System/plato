"""Callback dispatch must retain iteration and callback exception contracts."""

import pytest

from plato.callbacks.handler import CallbackHandler


def test_registered_callbacks_are_iterable_in_registration_order():
    class First:
        pass

    class Second:
        pass

    first, second = First(), Second()
    assert list(CallbackHandler([first, second])) == [first, second]


def test_callback_body_attribute_error_reaches_caller_unchanged():
    error = AttributeError("callback-sentinel")

    class Callback:
        def event(self):
            raise error

    handler = CallbackHandler([Callback()])
    with pytest.raises(AttributeError) as caught:
        handler.call_event("event")
    assert caught.value is error


def test_missing_callback_event_keeps_descriptive_error():
    handler = CallbackHandler([object()])
    with pytest.raises(ValueError, match="not been implemented"):
        handler.call_event("absent")
