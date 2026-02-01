@echo off
del "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Experts\GBPUSD_H1_QuadLayer_v69.ex5" 2>nul
"C:\Program Files\MetaTrader 5\metaeditor64.exe" /compile:"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Experts\GBPUSD_H1_QuadLayer_v69.mq5" /log
timeout /t 5 /nobreak
type "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Experts\GBPUSD_H1_QuadLayer_v69.log" 2>nul
