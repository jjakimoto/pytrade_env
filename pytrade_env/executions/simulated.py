from __future__ import print_function

try:
    import Queue as queue
except ImportError:
    import queue

from ..events import FillEvent
from .core import ExecutionHandler
from ..utils import get_time_now


class SimulatedExecutionHandler(ExecutionHandler):
    """
    The simulated execution handler simply converts all order
    objects into their equivalent fill objects automatically
    without latency, slippage or fill-ratio issues.

    This allows a straightforward "first go" test of any strategy,
    before implementation with a more sophisticated execution
    handler.
    """

    def __init__(self, events):
        """
        Initialises the handler, setting the event queues
        up internally.

        Parameters:
        events - The Queue of Event objects.
        """
        self.events = events

    def execute_order(self, event, commission=0):
        """
        Simply converts Order objects into Fill objects naively,
        i.e. without any latency, slippage or fill ratio problems.

        Parameters:
        event - Contains an Event object with order information.
        """
        if event.type == 'ORDER':
            fill_event = FillEvent(timeindex=get_time_now(),
                                   symbol=event.symbol,
                                   exchange='SIMULATION',
                                   quantity=event.quantity,
                                   direction=event.direction,
                                   fill_cost=None,
                                   commission=commission)
            self.events.put(fill_event)
