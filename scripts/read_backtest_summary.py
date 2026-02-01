"""Read MT5 backtest Excel report - Extract summary"""
import pandas as pd

file_path = r'C:\Users\Administrator\Documents\ReportTesterQuadV4-61045904.xlsx'

try:
    df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)

    # Find rows with key metrics
    keywords = ['profit', 'loss', 'balance', 'total', 'trades', 'drawdown', 'factor', 'expected', 'recovery', 'equity']

    print("="*60)
    print("BACKTEST SUMMARY - GBPUSD_H1_QuadLayer_v2")
    print("="*60)

    for idx, row in df.iterrows():
        row_text = ' '.join([str(x).lower() for x in row.values if pd.notna(x)])
        for kw in keywords:
            if kw in row_text:
                # Print this row
                values = [str(x) for x in row.values if pd.notna(x)]
                print(f"Row {idx}: {' | '.join(values)}")
                break

except Exception as e:
    print(f'Error: {e}')
