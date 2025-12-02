# Historical Data Setup

## Current Status

You have provided real historical monthly returns for 3 assets:
- ✅ **Bonds**: Corporate bond interest rate changes (1929-01 to 2025-10)
- ✅ **Gold**: Monthly % changes (1928-01 to 2025-10, mostly zeros until 1968)
- ✅ **US_Equities**: Large cap US equities (1927-12 to 2025-09)

## Still Needed

To run a full 6-asset optimization, you'll need:
- ❌ **Emerging_Markets** returns
- ❌ **Japan** returns  
- ❌ **Intl_Developed** returns

## Next Steps

###Option 1: Run 3-Asset Portfolio Now

I've updated your code to work with 3 assets. Simply:

1. The `historical_asset_returns.csv` file needs to be created from your pasted data
2. Run `portfolio_optimizer.py` with the 3-asset configuration
3. Get optimization results for Bonds, Gold, and US_Equities

### Option 2: Wait for All 6 Assets

Get the remaining 3 asset returns data, then we'll create a complete 6-asset CSV.

## Data Format

The CSV will have this structure:
```
Date,Bonds,Gold,US_Equities
1929-01-01,0.01,0.0,0.057
1929-02-01,0.04,0.0,-0.006
...
```

## How to Add More Assets Later

When you get the other 3 assets, just paste them and I'll:
1. Merge them into the existing CSV
2. Update the optimizer configuration
3. Re-run with all 6 assets


