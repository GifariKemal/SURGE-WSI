"""Read MT5 backtest Excel report"""
import pandas as pd
import sys

file_path = r'C:\Users\Administrator\Documents\ReportTesterQuadV2-61045904.xlsx'

try:
    xl = pd.ExcelFile(file_path)
    print('Sheets:', xl.sheet_names)

    for sheet in xl.sheet_names:
        print(f'\n{"="*60}')
        print(f'SHEET: {sheet}')
        print("="*60)
        df = pd.read_excel(file_path, sheet_name=sheet, header=None)
        # Print all rows
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        print(df.to_string())

except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
