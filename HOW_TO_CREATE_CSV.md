# How to Create historical_asset_returns.csv

## ✅ Code Status
- Your code is configured for **6 assets** (kept as-is, no changes)
- All your features intact: dynamic covariances, rebalancing, full metrics

## 📋 What You Need to Do

### Step 1: Open Excel or Google Sheets

### Step 2: Create 7 Columns with These Headers (EXACT ORDER):
```
Date | Emerging_Markets | Japan | Intl_Developed | Bonds | Gold | US_Equities
```

### Step 3: Fill in the Data

**For dates:** Start from 1929-01-01 and go monthly (1929-01-01, 1929-02-01, etc.)

**For the 3 assets you HAVE:**
- **Bonds column**: Paste your corporate bond returns
- **Gold column**: Paste your gold returns  
- **US_Equities column**: Paste your US equities returns

**For the 3 assets you DON'T have yet:**
- **Emerging_Markets column**: Put `0` in every row
- **Japan column**: Put `0` in every row
- **Intl_Developed column**: Put `0` in every row

### Step 4: Save
- File → Save As → CSV
- Name it: **historical_asset_returns.csv**
- Save in: `C:\Users\zisha\SynologyDrive\Drive\Zishaans_programs\WGHS\`

### Step 5: When You Get the Other 3 Assets Later
Just open the CSV and replace the zeros in those 3 columns. Done!

## 📊 Example of What the CSV Should Look Like:

```csv
Date,Emerging_Markets,Japan,Intl_Developed,Bonds,Gold,US_Equities
1929-01-01,0,0,0,0.01,0.0,0.057
1929-02-01,0,0,0,0.04,0.0,-0.006
1929-03-01,0,0,0,0.04,0.0,-0.002
...
```

## ⚠️ Important Notes

1. **Column order matters!** Must be exactly as shown above
2. **Dates must be YYYY-MM-DD format** (e.g., 1929-01-01)
3. **Returns are decimals** (0.01 = 1%, not "1%")
4. **No blank cells** - use 0.0 for missing data
5. **First row is headers** (Date, Emerging_Markets, etc.)

## 🚀 Once the CSV Exists

Simply run:
```bash
python portfolio_optimizer.py
```

The optimizer will:
- Load your CSV
- Use real data for Bonds/Gold/US_Equities
- Ignore the zero columns (they won't affect results)
- Run optimization when you have all 6 assets

## 💡 Alternative: I Can Create It For You

If you want, I can write a Python script that:
1. Takes your 3 pasted datasets
2. Parses them automatically
3. Creates the CSV with proper formatting
4. Adds the zero placeholders

Just say "create the script for me" and I'll do it!


