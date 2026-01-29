"""Check database structure"""
import asyncio
import sys
sys.path.append('.')
from config import config
import asyncpg

async def check_db():
    print("Connecting to database...")
    conn = await asyncpg.connect(config.database.connection_string)
    
    # Get tables
    tables = await conn.fetch('''
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public'
    ''')
    print("Tables:", [t['table_name'] for t in tables])
    
    # Get columns for ohlcv table
    for t in tables:
        tname = t['table_name']
        if 'ohlcv' in tname.lower():
            cols = await conn.fetch('''
                SELECT column_name, data_type FROM information_schema.columns 
                WHERE table_name = $1
            ''', tname)
            print(f"\nColumns in {tname}:")
            for c in cols:
                print(f"  - {c['column_name']}: {c['data_type']}")
            
            # Get sample row
            try:
                sample = await conn.fetchrow(f'SELECT * FROM {tname} LIMIT 1')
                if sample:
                    print(f"\nSample row:")
                    for k, v in dict(sample).items():
                        print(f"  {k}: {v}")
            except Exception as e:
                print(f"Error fetching sample: {e}")
    
    await conn.close()
    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(check_db())
