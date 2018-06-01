from __future__ import print_function

try:
    import Queue as queue
except ImportError:
    import queue
import numpy as np
import pandas as pd
from copy import deepcopy

from ..events import OrderEvent
from ..utils import create_sharpe_ratio, create_drawdowns


class Portfolio(object):
    """
    The Portfolio class handles the positions and market
    value of all instruments at a resolution of a "bar",
    i.e. secondly, minutely, 5-min, 30-min, 60 min or EOD.
    The positions DataFrame stores a time-index of the
    quantity of positions held.
    The holdings DataFrame stores the cash and total market
    holdings value of each symbol for a particular
    time-index, as well as the percentage change in
    portfolio total across bars.
    """

    def __init__(self, bars, events, initial_capital=1.0):
        """
        Initialises the portfolio with bars and an event queue.
        Also includes a starting datetime index and initial capital
        (USD unless otherwise stated).
        Parameters:
        bars - The DataHandler object with current market data.
        events - The Event Queue object.
        start_date - The start date (bar) of the portfolio.
        initial_capital - The starting capital in USD.
        """

        self.bars = bars
        self.events = events
        self.symbol_list = self.bars.symbol_list
        self.start = bars.start
        self.initial_capital = initial_capital

        # Position describes the size of each asset like the number of shares
        self.all_positions = self.construct_all_positions()
        self.current_positions = self.construct_current_positions()
        # Holding describes the value of each asset like according to USD
        self.all_holdings = self.construct_all_holdings()
        self.current_holdings = self.construct_current_holdings()

    def _initial_positions(self):
        return dict((s, 0) for s in self.symbol_list)

    def construct_all_positions(self):
        """
        Constructs the positions list using the start_date
        to determine when the time index will begin.
        """
        d = self._initial_positions()
        d['datetime'] = self.start
        return [d]

    def construct_current_positions(self):
        """
        Constructs the positions list using the start_date
        to determine when the time index will begin.
        """
        d = self._initial_positions()
        return d

    def construct_all_holdings(self):
        """
        Constructs the holdings list using the start_date
        to determine when the time index will begin.
        """
        d = self._initial_positions()
        d['datetime'] = self.start
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return [d]

    def construct_current_holdings(self):
        """
        This constructs the dictionary which will hold the instantaneous
        value of the portfolio across all symbols.
        """
        d = self._initial_positions()
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d

    def update_timeindex(self, event):
        """
        Adds a new record to the positions matrix for the current
        market data bar. This reflects the PREVIOUS bar, i.e. all
        current market data at this stage is known (OHLCV).
        Makes use of a MarketEvent from the events queue.
        """
        latest_datetime = self.bars.get_latest_bar_datetime(self.symbol_list[0])

        # Update positions
        dp = self._initial_positions()
        dp['datetime'] = latest_datetime

        for s in self.symbol_list:
            dp[s] = self.current_positions[s]

        # Append the current positions
        self.all_positions.append(dp)

        # Update holding
        dh = {}
        dh['datetime'] = latest_datetime
        dh['cash'] = self.current_holdings['cash']
        dh['commission'] = self.current_holdings['commission']
        dh['total'] = self.current_holdings['cash']

        for symbol in self.symbol_list:
            # Approximation to the real value
            market_value = self.current_positions[symbol] * \
                self.bars.get_latest_market_value(symbol)
            dh[symbol] = market_value
            dh['total'] += market_value

        self.current_holdings = deepcopy(dh)
        # Append the current holdings
        self.all_holdings.append(dh)

    def update_positions_from_fill(self, fill):
        """
        Takes a Fill object and updates the position matrix to
        reflect the new position.
        Parameters:
        fill - The Fill object to update the positions with.
        """
        # Check whether the fill is a buy or sell
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1
        # Update positions list with new quantities
        self.current_positions[fill.symbol] += fill_dir * fill.quantity

    def update_holdings_from_fill(self, fill):
        """
        Takes a Fill object and updates the holdings matrix to
        reflect the holdings value.

        Parameters:
        fill - The Fill object to update the holdings with.
        """
        # Check whether the fill is a buy or sell
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1

        # Update holdings list with new quantities
        if fill.fill_cost is None:
            fill_cost = self.bars.get_latest_market_value(fill.symbol)
            cost = fill_dir * fill_cost * fill.quantity
        else:
            cost = fill.fill_cost
        self.current_holdings[fill.symbol] += cost
        self.current_holdings['commission'] += fill.commission
        self.current_holdings['cash'] -= (cost + fill.commission)
        self.current_holdings['total'] -= fill.commission

    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings
        from a FillEvent.
        """
        if event.type == 'FILL':
            self.update_positions_from_fill(event)
            self.update_holdings_from_fill(event)

    def generate_order(self, signal):
        """
        Simply files an Order object as a constant quantity
        sizing of the signal object, without risk management or
        position sizing considerations.

        Parameters:
        signal - The tuple containing Signal information.
        """
        order = None
        symbol = signal.symbol
        mkt_quantity = self.get_quantity(symbol, signal.value)
        if mkt_quantity >= 0:
            signal.signal_type = 'LONG'
        else:
            signal.signal_type = 'SHORT'
        mkt_quantity = np.abs(mkt_quantity)
        direction = signal.signal_type
        cur_quantity = self.current_positions[symbol]
        order_type = 'MARKET'
        if direction == 'LONG':
            order = OrderEvent(symbol, order_type, mkt_quantity, 'BUY')
        if direction == 'SHORT':
            order = OrderEvent(symbol, order_type, mkt_quantity, 'SELL')
        if direction == 'EXIT' and cur_quantity > 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'SELL')
        if direction == 'EXIT' and cur_quantity < 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'BUY')
        return order

    def get_quantity(self, symbol, value):
        return value

    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new orders
        based on the portfolio logic.
        """
        if event.type == 'SIGNAL':
            # We should change here
            order_event = self.generate_order(event)
            self.events.put(order_event)

    def create_equity_curve_dataframe(self):
        """
        Creates a pandas DataFrame from the all_holdings
        list of dictionaries.
        """
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index('datetime', inplace=True)
        returns = curve['total'].pct_change()
        returns.values[0] = np.zeros_like(returns.values[0])
        curve['returns'] = returns
        curve['equity_curve'] = (1.0 + curve['returns']).cumprod()
        self.equity_curve = curve

    def output_summary_stats(self):
        """
        Creates a list of summary statistics for the portfolio.
        """
        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']
        sharpe_ratio = create_sharpe_ratio(returns, periods=252 * 60 * 6.5)
        drawdown, max_dd, dd_duration = create_drawdowns(pnl)
        self.equity_curve['drawdown'] = drawdown
        stats = [("Total Return", "%0.2f%%" % \
            ((total_return - 1.0) * 100.0)),
            ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
            ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
            ("Drawdown Duration", "%d" % dd_duration)]
        self.equity_curve.to_csv('equity.csv')
        return stats

    def get_stats(self):
        """
        Creates a list of summary statistics for the portfolio.
        """
        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']
        sharpe_ratio = create_sharpe_ratio(returns, periods=252 * 60 * 6.5)
        drawdown, max_dd, dd_duration = create_drawdowns(pnl)
        self.equity_curve['drawdown'] = drawdown
        stats = {"Total Return": (total_return - 1.0) * 100.0,
                 "Sharpe Ratio": sharpe_ratio,
                 "Max Drawdown": max_dd * 100.0,
                 "Drawdown Duration": dd_duration}
        return stats

    @property
    def weights(self):
        portfolio_values = self.portfolio_values
        weights = {}
        weights["cash"] = portfolio_values['cash'] / self.asset_size
        for key in portfolio_values.keys():
            weights[key] = portfolio_values[key] / self.asset_size
        return weights

    @property
    def weights_val(self):
        weights = self.weights
        weights_val = [weights["cash"]]
        for symbol in self.symbol_list:
            weights_val.append(weights[symbol])
        return np.array(weights_val)

    @property
    def portfolio_values(self):
        portfolio_values = {}
        portfolio_values['cash'] = self.current_holdings['cash']
        for symbol in self.symbol_list:
            portfolio_values[symbol] = self.current_holdings[symbol]
        return portfolio_values

    @property
    def asset_size(self):
        return np.sum(list(self.portfolio_values.values()))
