# Chrysos Lab

Testing grounds for prototype calculators, indicators, screeners, etc.

Each calculator is a plain Python module with typed inputs and outputs, kept apart from data
fetching and from any UI.

## Tools

```
uv run calendar-iv-crush AAPL    # pre-earnings calendar spread IV crush screen
uv run forward-factor AAPL       # rank calendar pairs (front 7-60 DTE, back 14-120 days later) by forward factor
uv run forward-factor --bulk ff.txt  # rank a watchlist's tickers by passing trades
```

Both take several symbols, prompt for symbols when given none, and print JSON with `--json`.

`--bulk` reads tickers from a text file (path relative to where you run the command), one per
line; blank lines and `#` comments are skipped. Tickers with no passing trade are left out, and
tickers that fail to fetch are listed under the table.
