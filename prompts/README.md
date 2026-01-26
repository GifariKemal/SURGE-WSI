# SURGE-WSI Illustration Prompts

Koleksi prompt untuk generate ilustrasi visual sistem SURGE-WSI menggunakan Nano Gemini atau AI image generator lainnya.

## Struktur Folder

```
prompts/illustrations/
├── general/           # 4 files - Overview & arsitektur
├── layer1_data/       # 4 files - Kalman Filter & Data Pipeline
├── layer2_regime/     # 3 files - HMM Regime Detection
├── layer3_killzone/   # 3 files - Kill Zone & Sessions
├── layer4_poi/        # 4 files - POI Detection (SMC)
├── layer5_entry/      # 4 files - Entry Trigger
├── layer6_risk/       # 3 files - Risk Management
├── layer7_exit/       # 3 files - Exit Management
├── integration/       # 3 files - Full System Integration
└── branding/          # 2 files - Logo & Colors
```

**Total: 33 prompt files**

## Format JSON

Setiap file menggunakan format berikut:

```json
{
  "id": "unique_id",
  "title": "Judul Ilustrasi",
  "category": "folder_category",
  "style": "technical_diagram|infographic|candlestick|ui_dashboard|minimalist",
  "prompt": "Prompt utama untuk image generation...",
  "negative_prompt": "Elements to avoid...",
  "aspect_ratio": "16:9|9:16|1:1",
  "color_scheme": ["#hex1", "#hex2", ...],
  "elements": ["element1", "element2", ...],
  "tags": ["tag1", "tag2", ...]
}
```

## Style Guide

| Kategori | Style | Keterangan |
|----------|-------|------------|
| General & Architecture | `technical_diagram` | Clean flowcharts, boxes, arrows |
| Layer Concepts | `infographic` | Colorful, icons, visual appeal |
| Trading (POI, Entry) | `candlestick` | Realistic trading chart visuals |
| Dashboards | `ui_dashboard` | Metrics, gauges, graphs |
| Branding | `minimalist` | Clean logo concepts |

## Color Palette

| Warna | Hex | Penggunaan |
|-------|-----|------------|
| Deep Navy | `#0F172A` | Background utama |
| Royal Blue | `#3B82F6` | Primary actions |
| Cyan | `#06B6D4` | Highlights |
| Bullish Green | `#10B981` | Profits, BUY |
| Bearish Red | `#EF4444` | Losses, SELL |
| Warning Yellow | `#FBBF24` | Caution |
| Sideways Gray | `#6B7280` | Neutral |
| Gold | `#F59E0B` | Premium |
| Purple | `#8B5CF6` | Regime |

## Cara Penggunaan

1. Pilih file JSON sesuai ilustrasi yang diinginkan
2. Copy nilai `prompt` ke Nano Gemini atau AI image generator
3. Gunakan `negative_prompt` jika supported
4. Set aspect ratio sesuai `aspect_ratio`
5. Adjust jika diperlukan berdasarkan hasil

## Contoh Penggunaan dengan Nano Gemini

```python
import json

# Load prompt
with open('prompts/illustrations/general/01_system_overview.json') as f:
    prompt_data = json.load(f)

# Generate image
image = generate_image(
    prompt=prompt_data['prompt'],
    negative_prompt=prompt_data['negative_prompt'],
    aspect_ratio=prompt_data['aspect_ratio']
)
```

## Daftar Lengkap Prompts

### General (4)
1. `01_system_overview.json` - SURGE-WSI hero illustration
2. `02_6layer_architecture.json` - 6-layer vertical flowchart
3. `03_data_flow_pipeline.json` - MT5 → DB → Redis → Kalman
4. `04_trading_workflow.json` - Complete decision tree

### Layer 1: Data Pipeline (4)
5. `01_kalman_filter_concept.json` - Noise reduction visualization
6. `02_mt5_connection.json` - MT5 ↔ Python bridge
7. `03_database_schema.json` - TimescaleDB ERD
8. `04_redis_cache.json` - Cache layer diagram

### Layer 2: Regime Detection (3)
9. `01_hmm_state_machine.json` - 3-state HMM diagram
10. `02_regime_probability.json` - Probability distribution
11. `03_confidence_meter.json` - Trading confidence gauge

### Layer 3: Kill Zone (3)
12. `01_session_calendar.json` - 24hr session wheel
13. `02_killzone_heatmap.json` - Weekly quality heatmap
14. `03_quality_score.json` - Score breakdown

### Layer 4: POI Detection (4)
15. `01_order_block_visual.json` - OB on candlestick
16. `02_fair_value_gap.json` - FVG visualization
17. `03_swing_points.json` - Swing highs/lows
18. `04_poi_lifecycle.json` - Detection → Mitigation

### Layer 5: Entry Trigger (4)
19. `01_liquidity_sweep.json` - Stop hunt pattern
20. `02_market_structure_shift.json` - MSS/CHOCH
21. `03_rejection_candle.json` - Wick rejection
22. `04_entry_quality_score.json` - Score components

### Layer 6: Risk Management (3)
23. `01_position_sizing.json` - Lot calculator
24. `02_risk_dashboard.json` - Risk limits gauges
25. `03_drawdown_scaling.json` - DD-based scaling

### Layer 7: Exit Management (3)
26. `01_partial_tp_waterfall.json` - TP1/TP2/TP3
27. `02_trailing_stop.json` - Trail mechanics
28. `03_position_lifecycle.json` - Entry → Exit timeline

### Integration (3)
29. `01_executor_flowchart.json` - Main execution flow
30. `02_telegram_mockup.json` - Bot message UI
31. `03_performance_dashboard.json` - Stats dashboard

### Branding (2)
32. `01_logo_concept.json` - SURIOTA/SURGE logo
33. `02_color_palette.json` - Brand colors

---

*Generated for SURGE-WSI by Claude AI*
