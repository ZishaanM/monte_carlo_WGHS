# How to Create historical_asset_returns.csv

## Quick Summary

Your `portfolio_optimizer.py` is now configured to work with **3 assets: Bonds, Gold, US_Equities**.

## Option 1: Use Excel/Google Sheets (Easiest)

1. **Open Excel or Google Sheets**
2. **Paste your three datasets into columns**:
   - Column A: Date
   - Column B: Bonds
   - Column C: Gold  
   - Column D: US_Equities

3. **Format the dates** as YYYY-MM-DD (e.g., 1929-01-01)

4. **Save as CSV**: File → Save As → CSV

5. **Name it**: `historical_asset_returns.csv`

6. **Place it in**: `C:\Users\zisha\SynologyDrive\Drive\Zishaans_programs\WGHS\`

### Example Format:
```
Date,Bonds,Gold,US_Equities
1929-01-01,0.01,0.0,0.057
1929-02-01,0.04,0.0,-0.006
1929-03-01,0.04,0.0,-0.002
...
```

##Option 2: Use Python Script (More Technical)

I can create a Python script that processes your pasted data if you'd like. Just say the word!

## Option 3: Run with Synthetic Data First (Test)

You can test the system with synthetic data first:

1. Open `portfolio_optimizer.py`
2. Change line 21 to: `USE_REAL_DATA = False`
3. Run: `python portfolio_optimizer.py`

This will run the optimizer with synthetic returns based on your model assumptions.

## What Happens Next

Once `historical_asset_returns.csv` exists:

1. Run: `python portfolio_optimizer.py`
2. It will:
   - Load your real historical data
   - Calculate dynamic covariances from the actual data
   - Run Monte Carlo simulations on all 3-asset allocation combinations
   - Save top strategies to `top_50_strategies.csv`

## Current Status

✅ Code updated for 3 assets (Bonds, Gold, US_Equities)
✅ Dynamic covariances enabled
✅ Rebalancing configured (hybrid quarterly with 5% threshold)
✅ All metrics included (Sharpe, Sortino, drawdowns, success rates)
⏳ Waiting for: `historical_asset_returns.csv` file

## When You Get More Assets

When you obtain data for Emerging Markets, Japan, and Intl Developed:

1. Add those columns to your CSV
2. Update `ASSET_NAMES` in `portfolio_optimizer.py`
3. Rerun the optimizer

That's it! I'll handle the rest automatically.


