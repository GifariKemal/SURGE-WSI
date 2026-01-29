import asyncio
import sys
sys.path.append('.')
from config import config
import asyncpg

async def main():
    conn = await asyncpg.connect(config.database.connection_string)
    rows = await conn.fetch('SELECT DISTINCT timeframe, COUNT(*) as cnt FROM ohlcv GROUP BY timeframe ORDER BY cnt DESC')
    print("Available timeframes in database:")
    for r in rows:
        tf = r['timeframe']
        cnt = r['cnt']
        print(f"  {tf}: {cnt:,} rows")
    
    # Also check date range
    minmax = await conn.fetchrow('SELECT MIN(time) as min_time, MAX(time) as max_time FROM ohlcv')
    print(f"\nDate range: {minmax['min_time']} to {minmax['max_time']}")
    
    await conn.close()

asyncio.run(main())
