 try:
    import Queue as queue
except ImportError:
    import queue
from abc import ABCMeta, abstractmethod
from tqdm import tqdm


class BaseRunner(object, metaclass=ABCMeta):
    def __init__(self, strategy, symbol_list, context, data_handler_cls,
                 execution_handler_cls, portfolio_cls):
        self.symbol_list = symbol_list
        self.strategy = strategy
        self.context = context
        self.keys = context.keys
        self.initial_capital = context.initial_capital
        self.commission_rate = context.commission_rate
        self.data_handler_cls = data_handler_cls
        self.execution_handler_cls = execution_handler_cls
        self.portfolio_cls = portfolio_cls

        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1

    def set_trange(self, start, end):
        self._start = start
        self._end = end

    def reset(self):
        self._generate_instances()

    def _generate_instances(self):
        # The count number of each event instances
        self.events = queue.Queue()
        self.data_handler = self.data_handler_cls(self.events,
                                                  self.symbol_list,
                                                  self.keys)
        self.data_handler.set_trange(self._start, self._end)
        self.start = self.data_handler.start
        self.end = self.data_handler.end
        self.portfolio = self.portfolio_cls(self.data_handler, self.events,
                                            self.start, self.initial_capital)
        self.execution_handler = self.execution_handler_cls(self.events)
        self.strategy.set(self.data_handler, self.events)

    def execute(self):
        while True:
            try:
                event = self.events.get(False)
            except queue.Empty:
                break
            else:
                if event is not None:
                    if event.type == 'MARKET':
                        self._calc_market(event)
                    elif event.type == 'SIGNAL':
                        self.signals += 1
                        self._calc_signal(event)
                    elif event.type == 'ORDER':
                        self.orders += 1
                        self._calc_order(event)
                    elif event.type == 'FILL':
                        self.fills += 1
                        self._calc_fill(event)
        self._update_strategy()

    def run(self, start, end):
        self.set_trange(start, end)
        self.reset()
        pbar = tqdm()
        try:
            while True:
                if self.data_handler.continue_trading:
                    self.prev_asset_size = self.portfolio.asset_size
                    self.data_handler.update_bars()
                    self.execute()
                    pbar.update(1)
                else:
                    break
        except KeyboardInterrupt:
            pass
        self.portfolio.create_equity_curve_dataframe()
        pbar.close()

    def output_summary_stats(self):
        return self.portfolio.output_summary_stats()

    @property
    def equity_curve(self):
        return self.portfolio.equity_curve

    @abstractmethod
    def _calc_market(self, event):
        raise NotImplementedError()

    @abstractmethod
    def _calc_signal(self, event):
        raise NotImplementedError()

    @abstractmethod
    def _calc_order(self, event):
        raise NotImplementedError()

    @abstractmethod
    def _calc_fill(self, event):
        raise NotImplementedError()

    @abstractmethod
    def _update_strategy(self):
        raise NotImplementedError()
