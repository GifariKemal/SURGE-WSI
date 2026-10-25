# CLAUDE.md - SURGE-WSI

Panduan kerja Claude Code di repo ini. `README.md` sebagian basi, lihat [Jebakan](#jebakan-yang-sudah-terverifikasi).

> [!CAUTION]
> Repo ini **PUBLIC** di GitHub. GitHub secret scanning sudah menandai Telegram
> bot token yang ter-commit di `strategies/gbpusd_h1_quadlayer/config.env`, dua
> file `.bat` di `strategies/rsi_v37_optimized/`, dan `.claude/settings.local.json`.
> Anggap token itu bocor: rotasi dan pembersihan history adalah keputusan owner,
> jangan dikerjakan tanpa instruksi eksplisit. Jangan buka atau print isinya.

## Apa ini

Sistem algo trading forex milik PT Surya Inovasi Prioritas (SURIOTA), Python plus
MetaTrader 5, fokus **GBPUSD H1**. Dua strategi aktif:

| Strategi | Inti | Entry point |
| --- | --- | --- |
| QuadLayer (v6.x, CHANGELOG terakhir v6.9.0) | Kalman + HMM regime + POI ICT/SMC (Order Block, FVG) di kill zone London/NY, lalu 4 layer quality filter: monthly profile, ATR/efficiency/ADX, intra-month risk, choppy detector | `strategies/gbpusd_h1_quadlayer/GBPUSD_H1_QuadLayer_Final/` |
| RSI v3.7 optimized | RSI mean reversion, filter regime SIDEWAYS + ConsecLoss3 | `strategies/rsi_v37_optimized/rsi_v37_optimized.py` (all-in-one) |

Port ke MQL5 (EA) dan PineScript dijaga sinkron dengan versi Python; banyak
commit adalah "sync" Python vs MQL5 vs backtest. Angka backtest di docstring dan
README adalah klaim lama, bukan hasil yang sudah diverifikasi ulang.

## Peta repo

| Path | Isi |
| --- | --- |
| `config/settings.yaml`, `config/__init__.py` | Parameter 6 layer (kalman, regime, killzone, poi, entry, exit, risk). Loader pydantic, `load_dotenv()` membaca `.env` di root |
| `src/analysis/` | `kalman_filter`, `regime_detector` (HMM), `poi_detector`, `market_filter` |
| `src/trading/` | Executor per versi (`executor_h1_v6_4_gbpusd.py` dipakai QuadLayer Final, `executor_rsi_v37*.py`), `risk_manager`, `exit_manager`, `adaptive_risk` |
| `src/utils/` | `telegram`, `news_filter`, `killzone`, `session_profiles`, aneka filter regime/volatility |
| `src/data/` | `mt5_connector.py` (tracked); `db_handler`, `cache`, `vector_*` hanya lokal, lihat Jebakan |
| `strategies/gbpusd_h1_quadlayer/` | `main.py` (bot v6.x), `executor.py`, `strategy_config.py`, filter, `experiments/` (riset), `CHANGELOG.md`, `QUICK_START.md` |
| `strategies/rsi_v37_optimized/` | Bot RSI v3.7 plus skrip analisis dan optimasi |
| `strategies/bbma/` | Riset BBMA (backtest saja) |
| `backtest/` | `backtester.py`, `monthly_backtest.py`, `h1_strategy/`, `archive/` |
| `mql5/` | `Experts/*.mq5` (QuadLayer v69, RSI v37), skrip backtest MT5 dan sweep |
| `pinescript/` | Versi TradingView QuadLayer v69 dan RSI v37 |
| `scripts/` | Sync data MT5 ke TimescaleDB, generator profil historis, debug Python vs MQL5 |
| `ml_trading_bot/` | Sub-proyek ML terpisah (punya README, requirements, `.env.example` sendiri) |
| `docker-compose.yml`, `docker/init-db.sql` | TimescaleDB, Redis, Qdrant, Adminer |
| `docs/` | Dokumen desain lama (`WORKFLOW.md`, `STRATEGY_COMPARISON.md`, `RSI_STRATEGY_EXPLAINED.md`, dll) |

## Setup dan menjalankan

Windows-only untuk apa pun yang menyentuh MT5 (paket `MetaTrader5`).

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
cp .env.example .env          # isi lokal saja, .env gitignored
docker-compose up -d          # TimescaleDB :5434, Redis :6381, Qdrant :6333, Adminer :8081
```

| Tugas | Perintah (dari root repo) |
| --- | --- |
| Backtest QuadLayer (data dari TimescaleDB) | `python strategies/gbpusd_h1_quadlayer/GBPUSD_H1_QuadLayer_Final/backtest_GBPUSD_H1_QuadLayer_Final.py` |
| Backtest multi-year QuadLayer (data langsung MT5) | `python strategies/gbpusd_h1_quadlayer/GBPUSD_H1_QuadLayer_Final/backtest_multi_year.py` |
| Backtest RSI v3.7 (data langsung MT5) | `python strategies/rsi_v37_optimized/backtest_rsi_v37_optimized.py` |
| Preflight sebelum demo | `python strategies/gbpusd_h1_quadlayer/GBPUSD_H1_QuadLayer_Final/preflight_checker.py` (cek Docker, MT5, DB, dan mengirim pesan test Telegram) |
| Bot demo QuadLayer | `python strategies/gbpusd_h1_quadlayer/GBPUSD_H1_QuadLayer_Final/GBPUSD_H1_QuadLayer_Final.py --demo --interval 300` (`--live` = akun live) |
| Bot RSI v3.7 | `python strategies/rsi_v37_optimized/rsi_v37_optimized.py` |
| Isi data DB | `scripts/sync_mt5_data.py`, `scripts/fetch_historical_data.py` (MT5 ke TimescaleDB) |
| Compile EA | `compile_ea.bat` (path MetaEditor dan terminal hardcoded ke PC owner) |

Test: `python -m pytest tests -q`. Unit test nyata ada di `test_executor`,
`test_kalman`, `test_poi`, `test_regime`, `test_trade_mode`. `test_config`,
`test_connection`, `test_mt5_login`, `test_mt5_simple` bukan pytest (0 fungsi
`test_`), itu skrip manual yang butuh terminal MT5 hidup. Belum ada CI.

## Data dan konfigurasi

- Sumber harga: terminal MT5 lokal. QuadLayer membaca OHLCV dari TimescaleDB
  (`DBHandler.get_ohlcv`), yang diisi skrip sync dari MT5. RSI dan multi-year
  memanggil `mt5.copy_rates_range` langsung.
- `data/`, `*.csv`, `*.parquet`, `*.pkl`, `models/`, `logs/` di-gitignore.
  `reports/` (PNG/PDF/XLSX hasil backtest, ribuan file market analysis) tracked.
- Akun yang dipakai bot QuadLayer: server **MetaQuotes-Demo**, bukan Finex
  (README strategi). Detail akun tidak pernah ditulis di repo.
- Dari environment (`.env.example`): `MT5_*` (login, password, server, terminal
  path), `POSTGRES_*`, `REDIS_*`, `TELEGRAM_BOT_TOKEN`/`_CHAT_ID`/`_ENABLED`,
  `TRADING_ENABLED`, `TRADING_MODE`, `SYMBOL`, `MAGIC_NUMBER`, `RISK_PER_TRADE`,
  `MAX_DAILY_LOSS`, `DAILY_PROFIT_TARGET`, `LOG_*`. Compose memakai `${...}`.

## Aturan secret (wajib)

- Secret (Telegram token, chat id, login/password broker, API key) **hanya** lewat
  environment variable atau `.env` root yang gitignored. Tidak pernah di
  `config.env`, `.bat`, `settings.yaml`, `.claude/*`, docstring, atau README.
- `config.env` dan `.bat` hanya boleh membawa placeholder kosong atau nilai
  non-rahasia. Launcher `.bat` membaca env, tidak meng-hardcode `set TOKEN=...`.
- `.claude/settings.local.json` harus tetap untracked (sudah di `.gitignore`).
- Global pre-commit secret scan (trufflehog di `~/.githooks`, via
  `core.hooksPath`) aktif. **Jangan pernah** bypass dengan `--no-verify`,
  `SKIP_SECRET_SCAN=1`, atau `git config hooks.secretscan false`. Kalau diblok,
  pindahkan nilainya ke `.env`, jangan akali hook-nya.
- Cek file rahasia hanya dengan `grep -c`/panjang, tidak pernah print nilainya.

## Jebakan yang sudah terverifikasi

- Pola `data/` di `.gitignore` ikut menelan `src/data/`: `db_handler.py`,
  `cache.py`, `vector_*.py`, `__init__.py` tidak ada di git. Clone baru gagal
  import `src.data.db_handler`. Perbaikan butuh keputusan owner (negasi pola).
- README dan `docs/LEGACY-CLAUDE.md` basi: `backtest.run_backtest`,
  `src.trading.executor --mode demo`, dan `config/settings.example.yaml` tidak ada.
- `strategies/gbpusd_h1_quadlayer/run.bat` memanggil `main_live.py` yang tidak
  ada (verify sebelum dipakai).
- Docstring versi tidak konsisten (main.py v6.6, file Final menyebut v6.4,
  CHANGELOG v6.9.0, EA v69). Rujukan utama perilaku: CHANGELOG plus kode.
- `.ex5` ter-commit walau `*.ex5` ada di `.gitignore` (force-add lama).

## Jangan

- Jangan jalankan bot dengan `--live`, kirim order, atau pakai `/test_buy`
  `/test_sell` tanpa instruksi eksplisit Gifari. Default selalu demo.
- Jangan commit `.env`, data, model, log, atau kredensial apa pun; jangan menulis
  nomor akun, chat id, atau IP ke file tracked.
- Jangan ubah parameter strategi di Python tanpa menyamakan EA MQL5 dan PineScript
  (atau catat jelas bahwa versinya kini beda).
- Jangan klaim edge dari backtest in-sample; sebut periode, biaya, dan sumber
  data. Jangan force-push atau rewrite history tanpa izin eksplisit.

## Gaya

Bahasa Indonesia natural bilingual, istilah trading tetap asli (candle, sweep,
drawdown, kill zone). ASCII saja, tanpa em-dash atau en-dash, kutip lurus.
Commit pakai conventional commits (`feat(mql5): ...`, `fix(rsi-v37): ...`).

Copyright (c) 2026 PT Surya Inovasi Prioritas (SURIOTA). Lisensi: lihat `LICENSE` (MIT).
