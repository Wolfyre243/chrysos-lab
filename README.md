# Chrysos Lab

Testing grounds for prototype calculators, indicators, screeners, etc.

Each calculator is a plain Python module with typed inputs and outputs, kept apart from data
fetching and from any UI.

## Tools

```
uv run calendar-iv-crush AAPL    # pre-earnings calendar spread IV crush screen
uv run forward-factor AAPL       # rank calendar pairs near 30/60 DTE by forward factor
```

Both take several symbols, prompt for symbols when given none, and print JSON with `--json`.
