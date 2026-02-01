"""Sync GBPUSD data from MetaQuotes to database"""
import asyncio
import asyncpg
import MetaTrader5 as mt5
from datetime import datetime, timezone

async def sync_metaquotes():
    # Connect to Metaquotes
    mt5.shutdown()
    if not mt5.initialize(path=r'C:\Program Files\MetaTrader 5\terminal64.exe'):
        print(f'Failed to connect to MT5: {mt5.last_error()}')
        return

    print('Connected to MetaQuotes-Demo')

    # Connect to database
    conn = await asyncpg.connect(
        host='localhost',
        port=5434,
        database='surge_wsi',
        user='surge_wsi',
        password='surge_wsi_secret'
    )

    # Disable limit
    await conn.execute('SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0')

    # Timeframes to sync
    timeframes = {
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
    }

    symbol = 'GBPUSD'
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2026, 2, 1)

    for tf_name, tf_val in timeframes.items():
        print(f'Syncing {symbol} {tf_name}...')

        rates = mt5.copy_rates_range(symbol, tf_val, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f'  No data for {tf_name}')
            continue

        print(f'  Downloaded {len(rates)} bars')

        # Prepare data for insert
        rows = []
        for r in rates:
            rows.append((
                datetime.fromtimestamp(r[0], tz=timezone.utc),
                symbol,
                tf_name,
                float(r[1]),  # open
                float(r[2]),  # high
                float(r[3]),  # low
                float(r[4]),  # close
                int(r[5]),    # tick_volume
                float(r[6]) if r[6] else 0,  # spread
                'metaquotes'  # source
            ))

        # Delete existing metaquotes data for this timeframe
        await conn.execute('''
            DELETE FROM ohlcv
            WHERE symbol = $1 AND timeframe = $2 AND source = 'metaquotes'
        ''', symbol, tf_name)

        # Insert new data
        await conn.executemany('''
            INSERT INTO ohlcv (time, symbol, timeframe, open, high, low, close, volume, spread, source)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (time, symbol, timeframe) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                spread = EXCLUDED.spread,
                source = EXCLUDED.source
        ''', rows)

        print(f'  Inserted {len(rows)} bars')

    # Show final summary
    summary = await conn.fetch('''
        SELECT source, timeframe, COUNT(*) as bars,
               MIN(time) as first_bar, MAX(time) as last_bar
        FROM ohlcv
        WHERE symbol = 'GBPUSD'
        GROUP BY source, timeframe
        ORDER BY source, timeframe
    ''')

    print('')
    print('=' * 70)
    print('FINAL DATA SUMMARY')
    print('=' * 70)
    for row in summary:
        print(f"{row['source']:12} | {row['timeframe']:5} | {row['bars']:>8,} bars | {str(row['first_bar'])[:10]} to {str(row['last_bar'])[:10]}")

    await conn.close()
    mt5.shutdown()
    print('')
    print('Sync completed!')

if __name__ == '__main__':
    asyncio.run(sync_metaquotes())
