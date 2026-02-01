//+------------------------------------------------------------------+
//|                                     GBPUSD_H1_QuadLayer_v3.mq5   |
//|                                    SURGE-WSI Trading System      |
//|                                    Version 3.0 - Python Sync     |
//+------------------------------------------------------------------+
//| CHANGES FROM v2:                                                  |
//| 1. Fixed pipSize calculation (was 10x too large)                  |
//| 2. Added Entry Multipliers (MOMENTUM/ENGULF/LOWER_HIGH)           |
//| 3. Added risk multiplier clamping (0.30 - 1.20)                   |
//| 4. Changed SL_CAPPED to post-trade cap (like Python)              |
//| 5. Synced all multiplier values with Python backtest              |
//+------------------------------------------------------------------+
#property copyright "SURGE-WSI"
#property link      "https://github.com/surge-wsi"
#property version   "3.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>

//+------------------------------------------------------------------+
//| Input Parameters                                                  |
//+------------------------------------------------------------------+
input group "=== Risk Management ==="
input double   RiskPercent = 1.0;           // Risk per trade (%)
input double   MaxLossPercent = 0.15;       // Max Loss per trade (%) - POST-TRADE CAP
input double   SL_ATR_Mult = 1.5;           // SL ATR Multiplier
input double   TP_Ratio = 1.5;              // TP:SL Ratio
input double   MaxLotSize = 5.0;            // Maximum Lot Size
input double   MinLotSize = 0.01;           // Minimum Lot Size

input group "=== ATR Settings ==="
input int      ATR_Period = 14;             // ATR Period
input double   MinATR_Pips = 8.0;           // Minimum ATR (pips)
input double   MaxATR_Pips = 25.0;          // Maximum ATR (pips)

input group "=== EMA Settings ==="
input int      EMA_Fast = 20;               // Fast EMA Period
input int      EMA_Slow = 50;               // Slow EMA Period

input group "=== Entry Signals ==="
input bool     UseOrderBlock = true;        // Use Order Block Entry
input bool     UseEmaPullback = true;       // Use EMA Pullback Entry
input double   MinQuality = 60.0;           // Minimum Signal Quality
input bool     UseEntryTrigger = true;      // Require Entry Trigger (MOMENTUM/ENGULF/LOWER_HIGH)

input group "=== Layer 3: Intra-Month Risk ==="
input double   MonthlyLossStop = -400.0;    // Monthly loss circuit breaker ($)
input int      ConsecLossMax = 6;           // Max consecutive losses before day stop

input group "=== Layer 4: Pattern Filter ==="
input bool     UsePatternFilter = true;     // Enable Pattern-Based Filter
input int      WarmupTrades = 15;           // Warmup trades (observe only)
input double   RollingWRHalt = 0.10;        // Rolling WR threshold for halt
input int      RollingWindow = 10;          // Rolling window size

input group "=== Session Filter ==="
input bool     UseSessionFilter = true;     // Enable Session Filter
input bool     SkipHour11 = true;           // Skip Hour 11 (low WR)
input bool     SkipOB_Hour8 = true;         // Skip OrderBlock at Hour 8
input bool     SkipOB_Hour16 = true;        // Skip OrderBlock at Hour 16
input bool     SkipEMA_Hour13 = true;       // Skip EMA Pullback Hour 13
input bool     SkipEMA_Hour14 = true;       // Skip EMA Pullback Hour 14

input group "=== Trading Hours (UTC) ==="
input bool     BacktestUTC = true;          // [BACKTEST] Data is already UTC
input bool     AutoDetectGMT = true;        // [LIVE] Auto-detect Broker GMT Offset
input int      GMTOffset = 0;               // [LIVE] Manual GMT Offset
input int      LondonStart = 8;             // London Session Start (UTC)
input int      LondonEnd = 10;              // London Session End (UTC)
input int      NewYorkStart = 13;           // New York Session Start (UTC)
input int      NewYorkEnd = 17;             // New York Session End

input group "=== Magic Number ==="
input int      MagicNumber = 69003;         // EA Magic Number (v3)

input group "=== Debug ==="
input bool     DebugMode = false;           // Enable Debug Logging

//+------------------------------------------------------------------+
//| Global Variables                                                  |
//+------------------------------------------------------------------+
CTrade         trade;
CPositionInfo  posInfo;
CAccountInfo   accInfo;

int            atrHandle;
int            emaFastHandle;
int            emaSlowHandle;
int            rsiHandle;
int            adxHandle;

// v3 FIX: Correct pip values matching Python
double         pipSize;       // 0.0001 for GBPUSD (NOT multiplied by 10!)
double         pipValue;      // ~10 USD per pip per standard lot
int            detectedGMTOffset;

// Day Multipliers (Python: 0=Mon, 4=Fri) -> MQL5: 0=Sun, 1=Mon, ..., 6=Sat
// Python values: Mon=1.0, Tue=0.9, Wed=1.0, Thu=0.8, Fri=0.3, Sat/Sun=0.0
double DayMultipliers[7] = {0.0, 1.0, 0.9, 1.0, 0.8, 0.3, 0.0};

// Hour Multipliers (UTC) - exact match with Python
double HourMultipliers[24] = {
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // 0-5
   0.5, 0.0, 1.0, 1.0, 0.9, 0.0,  // 6-11 (7,11 = skip)
   0.7, 1.0, 1.0, 1.0, 0.9, 0.7,  // 12-17
   0.3, 0.0, 0.0, 0.0, 0.0, 0.0   // 18-23
};

// v3 NEW: Entry Multipliers (matching Python ENTRY_MULTIPLIERS)
// MOMENTUM=1.0, LOWER_HIGH=1.0, ENGULF=0.8, DIRECT=1.0
double GetEntryMultiplier(string entryType)
{
   if(entryType == "MOMENTUM") return 1.0;
   if(entryType == "LOWER_HIGH") return 1.0;
   if(entryType == "ENGULF") return 0.8;
   return 1.0;  // DIRECT or unknown
}

// Monthly Tradeable Percentage (from Python SEASONAL_TEMPLATE)
double MonthlyTradeablePct[12] = {
   65,  // Jan
   55,  // Feb - POOR!
   70,  // Mar
   70,  // Apr
   62,  // May
   68,  // Jun
   78,  // Jul
   65,  // Aug
   72,  // Sep
   58,  // Oct
   66,  // Nov
   60   // Dec
};

// Monthly Risk Multipliers (from Python MONTHLY_RISK)
double MonthlyRiskMult[12] = {
   0.9,   // Jan
   0.6,   // Feb - lowered
   0.8,   // Mar
   1.0,   // Apr
   0.7,   // May
   0.85,  // Jun
   1.0,   // Jul
   0.75,  // Aug
   0.9,   // Sep
   0.6,   // Oct
   0.75,  // Nov
   0.8    // Dec
};

// Layer 2: Technical thresholds
#define ATR_STABILITY_THRESHOLD    0.25
#define EFFICIENCY_THRESHOLD       0.08
#define TREND_STRENGTH_THRESHOLD   25.0
#define BASE_QUALITY_GOOD          60
#define BASE_QUALITY_NORMAL        65
#define BASE_QUALITY_BAD           80

// Layer 3: Intra-Month Risk
int    currentMonth = 0;
int    currentDay = 0;
double monthlyPnL = 0.0;
int    consecutiveLosses = 0;
bool   monthStopped = false;
bool   dayStopped = false;

// Layer 4: Pattern Filter
struct TradeRecord {
   string direction;
   double pnl;
   datetime time;
};
TradeRecord tradeHistory[50];
int         tradeHistoryCount = 0;
bool        patternHalted = false;
bool        inRecovery = false;
int         recoveryWins = 0;

// v3: Track entry type for current signal
string      currentEntryType = "";

//+------------------------------------------------------------------+
//| Check if running in Strategy Tester                               |
//+------------------------------------------------------------------+
bool IsBacktest()
{
   return MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_OPTIMIZATION);
}

//+------------------------------------------------------------------+
//| LAYER 1: Get Monthly Quality Adjustment                           |
//+------------------------------------------------------------------+
int GetMonthlyQualityAdjustment(int month)
{
   double tradeable = MonthlyTradeablePct[month - 1];

   if(tradeable < 30)      return 50;
   else if(tradeable < 40) return 35;
   else if(tradeable < 50) return 25;
   else if(tradeable < 60) return 15;
   else if(tradeable < 70) return 10;
   else if(tradeable < 75) return 5;
   else                    return 0;
}

//+------------------------------------------------------------------+
//| LAYER 1: Get Monthly Risk Multiplier                              |
//+------------------------------------------------------------------+
double GetMonthlyRiskMult(int month)
{
   return MonthlyRiskMult[month - 1];
}

//+------------------------------------------------------------------+
//| LAYER 3: Check Intra-Month Risk Manager                           |
//+------------------------------------------------------------------+
int CheckIntraMonthRisk(datetime currentTime, int &dynamicAdj)
{
   MqlDateTime dt;
   TimeToStruct(currentTime, dt);

   int monthKey = dt.year * 100 + dt.mon;
   int dayKey = dt.year * 10000 + dt.mon * 100 + dt.day;

   if(monthKey != currentMonth)
   {
      currentMonth = monthKey;
      monthlyPnL = 0.0;
      consecutiveLosses = 0;
      monthStopped = false;
      dayStopped = false;
   }

   if(dayKey != currentDay)
   {
      currentDay = dayKey;
      dayStopped = false;
   }

   if(monthStopped) return 1;
   if(dayStopped) return 2;

   dynamicAdj = 0;

   if(monthlyPnL <= MonthlyLossStop)
   {
      monthStopped = true;
      Print("LAYER3: Month stopped at P&L $", monthlyPnL);
      return 1;
   }

   if(consecutiveLosses >= ConsecLossMax)
   {
      dayStopped = true;
      Print("LAYER3: Day stopped after ", consecutiveLosses, " consecutive losses");
      return 2;
   }

   // Dynamic quality adjustments
   if(monthlyPnL <= -350)      dynamicAdj = 15;
   else if(monthlyPnL <= -250) dynamicAdj = 10;
   else if(monthlyPnL <= -150) dynamicAdj = 5;

   if(consecutiveLosses >= 3)  dynamicAdj += 5;

   return 0;
}

//+------------------------------------------------------------------+
//| LAYER 3: Record Trade for Risk Manager                            |
//+------------------------------------------------------------------+
void RecordTradeForRiskManager(double pnl)
{
   monthlyPnL += pnl;

   if(pnl < 0)
      consecutiveLosses++;
   else
      consecutiveLosses = 0;

   if(DebugMode)
      Print("LAYER3: Monthly P&L = $", monthlyPnL, ", Consecutive Losses = ", consecutiveLosses);
}

//+------------------------------------------------------------------+
//| LAYER 4: Get Rolling Win Rate                                     |
//+------------------------------------------------------------------+
double GetRollingWinRate()
{
   if(tradeHistoryCount < 3) return 1.0;

   int lookback = MathMin(RollingWindow, tradeHistoryCount);
   int wins = 0;

   for(int i = tradeHistoryCount - lookback; i < tradeHistoryCount; i++)
   {
      if(tradeHistory[i].pnl > 0) wins++;
   }

   return (double)wins / (double)lookback;
}

//+------------------------------------------------------------------+
//| LAYER 4: Check Both Directions Failing                            |
//+------------------------------------------------------------------+
bool AreBothDirectionsFailing()
{
   if(tradeHistoryCount < 8) return false;

   int lookback = MathMin(8, tradeHistoryCount);
   int buyLosses = 0, sellLosses = 0;

   for(int i = tradeHistoryCount - lookback; i < tradeHistoryCount; i++)
   {
      if(tradeHistory[i].pnl < 0)
      {
         if(tradeHistory[i].direction == "BUY") buyLosses++;
         else sellLosses++;
      }
   }

   return (buyLosses >= 4 && sellLosses >= 4);
}

//+------------------------------------------------------------------+
//| LAYER 4: Check Pattern Filter                                     |
//+------------------------------------------------------------------+
bool CheckPatternFilter(string direction, double &sizeMult, int &extraQuality)
{
   sizeMult = 1.0;
   extraQuality = 0;

   if(!UsePatternFilter) return true;
   if(tradeHistoryCount < WarmupTrades) return true;

   if(patternHalted && !inRecovery) return false;

   if(AreBothDirectionsFailing())
   {
      patternHalted = true;
      inRecovery = true;
      recoveryWins = 0;
      Print("LAYER4: HALT - Both directions failing");
      return false;
   }

   double rollingWR = GetRollingWinRate();
   if(rollingWR < RollingWRHalt)
   {
      patternHalted = true;
      inRecovery = true;
      recoveryWins = 0;
      Print("LAYER4: HALT - Rolling WR too low: ", rollingWR * 100, "%");
      return false;
   }

   if(inRecovery)
   {
      sizeMult = 0.5;
      extraQuality = 5;
   }
   else if(rollingWR < 0.25)
   {
      sizeMult = 0.6;
      extraQuality = 3;
   }

   return true;
}

//+------------------------------------------------------------------+
//| LAYER 4: Record Trade for Pattern Filter                          |
//+------------------------------------------------------------------+
void RecordTradeForPatternFilter(string direction, double pnl, datetime time)
{
   if(!UsePatternFilter) return;

   if(tradeHistoryCount < 50)
   {
      tradeHistory[tradeHistoryCount].direction = direction;
      tradeHistory[tradeHistoryCount].pnl = pnl;
      tradeHistory[tradeHistoryCount].time = time;
      tradeHistoryCount++;
   }
   else
   {
      for(int i = 0; i < 49; i++)
         tradeHistory[i] = tradeHistory[i+1];
      tradeHistory[49].direction = direction;
      tradeHistory[49].pnl = pnl;
      tradeHistory[49].time = time;
   }

   if(inRecovery)
   {
      if(pnl > 0)
      {
         recoveryWins++;
         if(recoveryWins >= 1)
         {
            patternHalted = false;
            inRecovery = false;
            recoveryWins = 0;
            Print("LAYER4: Recovery complete");
         }
      }
      else
      {
         recoveryWins = 0;
      }
   }
}

//+------------------------------------------------------------------+
//| Check Entry Trigger                                               |
//+------------------------------------------------------------------+
bool CheckEntryTrigger(int direction, string &entryType)
{
   if(!UseEntryTrigger)
   {
      entryType = "DIRECT";
      return true;
   }

   double open1 = iOpen(_Symbol, PERIOD_H1, 1);
   double close1 = iClose(_Symbol, PERIOD_H1, 1);
   double high1 = iHigh(_Symbol, PERIOD_H1, 1);
   double low1 = iLow(_Symbol, PERIOD_H1, 1);

   double open2 = iOpen(_Symbol, PERIOD_H1, 2);
   double close2 = iClose(_Symbol, PERIOD_H1, 2);
   double high2 = iHigh(_Symbol, PERIOD_H1, 2);

   double range1 = high1 - low1;
   if(range1 < 0.0003) return false;

   double body1 = MathAbs(close1 - open1);
   double prevBody = MathAbs(close2 - open2);
   bool isBullish = close1 > open1;
   bool isBearish = close1 < open1;
   bool prevBullish = close2 > open2;
   bool prevBearish = close2 < open2;

   // MOMENTUM: Strong candle
   if(body1 > range1 * 0.5)
   {
      if(direction > 0 && isBullish)
      {
         entryType = "MOMENTUM";
         return true;
      }
      if(direction < 0 && isBearish)
      {
         entryType = "MOMENTUM";
         return true;
      }
   }

   // ENGULF: Engulfing pattern
   if(body1 > prevBody * 1.2)
   {
      if(direction > 0 && isBullish && prevBearish)
      {
         entryType = "ENGULF";
         return true;
      }
      if(direction < 0 && isBearish && prevBullish)
      {
         entryType = "ENGULF";
         return true;
      }
   }

   // LOWER_HIGH: For SELL
   if(direction < 0 && high1 < high2 && isBearish)
   {
      entryType = "LOWER_HIGH";
      return true;
   }

   return false;
}

//+------------------------------------------------------------------+
//| LAYER 2: Assess Technical Market Condition                        |
//+------------------------------------------------------------------+
int AssessTechnicalCondition(string &marketLabel)
{
   int lookback = 20;

   double atrValues[];
   ArraySetAsSeries(atrValues, true);
   if(CopyBuffer(atrHandle, 0, 1, lookback, atrValues) < lookback)
   {
      marketLabel = "NO_DATA";
      return BASE_QUALITY_NORMAL;
   }

   double atrSum = 0, atrSumSq = 0;
   for(int i = 0; i < lookback; i++)
   {
      atrSum += atrValues[i];
      atrSumSq += atrValues[i] * atrValues[i];
   }
   double atrMean = atrSum / lookback;
   double atrVariance = (atrSumSq / lookback) - (atrMean * atrMean);
   double atrStd = MathSqrt(MathMax(0, atrVariance));
   double atrCV = (atrMean > 0) ? atrStd / atrMean : 0.5;

   double closes[];
   ArraySetAsSeries(closes, true);
   if(CopyClose(_Symbol, PERIOD_H1, 1, lookback + 1, closes) < lookback + 1)
   {
      marketLabel = "NO_DATA";
      return BASE_QUALITY_NORMAL;
   }

   double netMove = MathAbs(closes[0] - closes[lookback]);
   double totalMove = 0;
   for(int i = 0; i < lookback; i++)
   {
      totalMove += MathAbs(closes[i] - closes[i + 1]);
   }
   double efficiency = (totalMove > 0) ? netMove / totalMove : 0.1;

   double adxValues[];
   ArraySetAsSeries(adxValues, true);
   if(CopyBuffer(adxHandle, 0, 1, 3, adxValues) < 3)
   {
      marketLabel = "NO_DATA";
      return BASE_QUALITY_NORMAL;
   }
   double trendStrength = adxValues[0];

   int score = 0;

   if(atrCV < ATR_STABILITY_THRESHOLD) score += 33;
   else if(atrCV < ATR_STABILITY_THRESHOLD * 1.5) score += 20;

   if(efficiency > EFFICIENCY_THRESHOLD) score += 33;
   else if(efficiency > EFFICIENCY_THRESHOLD * 0.5) score += 20;

   if(trendStrength > TREND_STRENGTH_THRESHOLD) score += 34;
   else if(trendStrength > TREND_STRENGTH_THRESHOLD * 0.7) score += 20;

   int technicalQuality;
   if(score >= 80)
   {
      technicalQuality = BASE_QUALITY_GOOD;
      marketLabel = "GOOD";
   }
   else if(score >= 40)
   {
      technicalQuality = BASE_QUALITY_NORMAL;
      marketLabel = "NORMAL";
   }
   else
   {
      technicalQuality = BASE_QUALITY_BAD;
      marketLabel = "BAD";
   }

   if(DebugMode)
      Print("LAYER2: ATR_CV=", DoubleToString(atrCV, 3),
            " Efficiency=", DoubleToString(efficiency, 3),
            " ADX=", DoubleToString(trendStrength, 1),
            " Score=", score, " -> ", marketLabel);

   return technicalQuality;
}

//+------------------------------------------------------------------+
//| Detect broker GMT offset (LIVE TRADING ONLY)                      |
//+------------------------------------------------------------------+
int DetectGMTOffset()
{
   datetime gmtTime = TimeGMT();
   datetime serverTime = TimeCurrent();

   int diffSeconds = (int)(serverTime - gmtTime);
   int diffHours = diffSeconds / 3600;

   if(diffSeconds % 3600 > 1800) diffHours += 1;
   else if(diffSeconds % 3600 < -1800) diffHours -= 1;

   return diffHours;
}

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_IOC);

   // GMT Offset Logic
   if(IsBacktest())
   {
      if(BacktestUTC)
      {
         detectedGMTOffset = 0;
         Print("=== BACKTEST MODE (v3) ===");
         Print("BacktestUTC=true: Data is already UTC, GMTOffset=0");
      }
      else
      {
         detectedGMTOffset = GMTOffset;
         Print("=== BACKTEST MODE (LEGACY) ===");
         Print("Using manual GMTOffset=", GMTOffset);
      }
   }
   else
   {
      if(AutoDetectGMT)
      {
         detectedGMTOffset = DetectGMTOffset();
         Print("=== LIVE MODE ===");
         Print("GMT Offset AUTO-DETECTED: GMT+", detectedGMTOffset);
      }
      else
      {
         detectedGMTOffset = GMTOffset;
         Print("=== LIVE MODE ===");
         Print("GMT Offset MANUAL: GMT+", detectedGMTOffset);
      }
   }

   // v3 FIX: Correct pip size calculation
   // SYMBOL_POINT returns 0.00001 for 5-digit broker or 0.0001 for 4-digit
   // For GBPUSD, 1 pip = 0.0001, so we need to adjust for 5-digit brokers
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   if(digits == 5 || digits == 3)  // 5-digit broker (or 3-digit for JPY)
      pipSize = point * 10;  // 0.00001 * 10 = 0.0001
   else
      pipSize = point;        // 4-digit broker: 0.0001

   // pipValue = value per pip per 1.0 lot
   // For GBPUSD with standard lot (100,000 units), 1 pip = $10
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

   // Calculate pip value: tickValue * (pipSize / tickSize)
   if(tickSize > 0)
      pipValue = tickValue * (pipSize / tickSize);
   else
      pipValue = 10.0;  // Fallback to standard

   Print("=== PIP CALCULATION (v3 FIXED) ===");
   Print("Point=", point, " Digits=", digits);
   Print("pipSize=", pipSize, " (should be 0.0001 for GBPUSD)");
   Print("pipValue=", pipValue, " (should be ~10 USD per lot)");

   // Initialize indicators
   atrHandle = iATR(_Symbol, PERIOD_H1, ATR_Period);
   emaFastHandle = iMA(_Symbol, PERIOD_H1, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   emaSlowHandle = iMA(_Symbol, PERIOD_H1, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
   rsiHandle = iRSI(_Symbol, PERIOD_H1, 14, PRICE_CLOSE);
   adxHandle = iADX(_Symbol, PERIOD_H1, 14);

   if(atrHandle == INVALID_HANDLE || emaFastHandle == INVALID_HANDLE ||
      emaSlowHandle == INVALID_HANDLE || rsiHandle == INVALID_HANDLE ||
      adxHandle == INVALID_HANDLE)
   {
      Print("Error initializing indicators!");
      return INIT_FAILED;
   }

   Print("=== GBPUSD H1 QuadLayer v3.0 (Python Sync) ===");
   Print("Risk: ", RiskPercent, "% | SL: ", SL_ATR_Mult, "x ATR | TP: ", TP_Ratio, ":1");
   Print("MaxLossPercent: ", MaxLossPercent, "% (POST-TRADE CAP like Python)");
   Print("Entry Multipliers: MOMENTUM=1.0, ENGULF=0.8, LOWER_HIGH=1.0");
   Print("Risk Clamp: min=0.30, max=1.20 (matching Python)");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(atrHandle);
   IndicatorRelease(emaFastHandle);
   IndicatorRelease(emaSlowHandle);
   IndicatorRelease(rsiHandle);
   IndicatorRelease(adxHandle);
   Print("GBPUSD H1 QuadLayer v3.0 deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   static datetime lastBar = 0;
   static datetime firstBar = 0;
   static int barCount = 0;

   datetime currentBar = iTime(_Symbol, PERIOD_H1, 0);
   if(lastBar == currentBar) return;
   lastBar = currentBar;

   if(firstBar == 0)
   {
      firstBar = currentBar;
      barCount = 0;
   }
   barCount++;

   // 100-bar warmup
   if(barCount < 100)
   {
      if(DebugMode) Print("DEBUG: Warmup - bar ", barCount, "/100");
      return;
   }

   if(HasOpenPosition()) return;

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);

   int utcHour = dt.hour - detectedGMTOffset;
   if(utcHour < 0) utcHour += 24;
   if(utcHour >= 24) utcHour -= 24;

   // Weekend filter
   if(dt.day_of_week == 0 || dt.day_of_week == 6)
   {
      if(DebugMode) Print("DEBUG: Skipped - Weekend");
      return;
   }

   double dayMult = DayMultipliers[dt.day_of_week];
   if(dayMult <= 0.0)
   {
      if(DebugMode) Print("DEBUG: Skipped - DayMult=0");
      return;
   }

   double hourMult = HourMultipliers[utcHour];
   if(hourMult <= 0.0)
   {
      if(DebugMode) Print("DEBUG: Skipped - HourMult=0 (UTC Hour=", utcHour, ")");
      return;
   }

   // Kill zone check
   bool inLondon = (utcHour >= LondonStart && utcHour <= LondonEnd);
   bool inNewYork = (utcHour >= NewYorkStart && utcHour <= NewYorkEnd);
   if(!inLondon && !inNewYork)
   {
      if(DebugMode) Print("DEBUG: Skipped - Outside Kill Zone");
      return;
   }

   // Skip Hour 11
   if(SkipHour11 && utcHour == 11) return;

   // LAYER 3: Intra-Month Risk
   int dynamicAdj = 0;
   int riskCheck = CheckIntraMonthRisk(TimeCurrent(), dynamicAdj);
   if(riskCheck == 1 || riskCheck == 2) return;

   // Get indicators
   double atr[], emaFast[], emaSlow[], rsi[], adxMain[];
   ArraySetAsSeries(atr, true);
   ArraySetAsSeries(emaFast, true);
   ArraySetAsSeries(emaSlow, true);
   ArraySetAsSeries(rsi, true);
   ArraySetAsSeries(adxMain, true);

   if(CopyBuffer(atrHandle, 0, 0, 3, atr) < 3) return;
   if(CopyBuffer(emaFastHandle, 0, 0, 3, emaFast) < 3) return;
   if(CopyBuffer(emaSlowHandle, 0, 0, 3, emaSlow) < 3) return;
   if(CopyBuffer(rsiHandle, 0, 0, 3, rsi) < 3) return;
   if(CopyBuffer(adxHandle, 0, 0, 3, adxMain) < 3) return;

   // v3 FIX: Correct ATR to pips conversion
   double atrPips = atr[1] / pipSize;

   if(atrPips < MinATR_Pips || atrPips > MaxATR_Pips)
   {
      if(DebugMode) Print("DEBUG: Skipped - ATR out of range (", DoubleToString(atrPips, 1), " pips)");
      return;
   }

   // Detect regime
   double close = iClose(_Symbol, PERIOD_H1, 1);
   int regime = 0;

   if(close > emaFast[1] && emaFast[1] > emaSlow[1])
      regime = 1;
   else if(close < emaFast[1] && emaFast[1] < emaSlow[1])
      regime = -1;

   if(regime == 0) return;

   // LAYER 2: Technical Assessment
   string marketLabel = "";
   int technicalQuality = AssessTechnicalCondition(marketLabel);

   // LAYER 1: Monthly Quality Adjustment
   int monthlyAdj = GetMonthlyQualityAdjustment(dt.mon);
   double effectiveMinQuality = technicalQuality + monthlyAdj + dynamicAdj;

   // Signal detection
   int signal = 0;
   string signalType = "";
   double quality = 0;

   // EMA Pullback first
   if(UseEmaPullback)
   {
      bool skipEMA = false;
      if(UseSessionFilter)
      {
         if(SkipEMA_Hour13 && utcHour == 13) skipEMA = true;
         if(SkipEMA_Hour14 && utcHour == 14) skipEMA = true;
      }

      if(!skipEMA)
      {
         int emaSignal = CheckEmaPullback(emaFast, emaSlow, rsi, adxMain, atrPips, quality);
         if(emaSignal != 0 && quality >= effectiveMinQuality)
         {
            if((emaSignal > 0 && regime > 0) || (emaSignal < 0 && regime < 0))
            {
               signal = emaSignal;
               signalType = "EMA_PULLBACK";
            }
         }
      }
   }

   // Order Block fallback
   if(signal == 0 && UseOrderBlock)
   {
      bool skipOB = false;
      if(UseSessionFilter)
      {
         if(SkipOB_Hour8 && utcHour == 8) skipOB = true;
         if(SkipOB_Hour16 && utcHour == 16) skipOB = true;
      }

      if(!skipOB)
      {
         int obSignal = CheckOrderBlock(quality);
         if(obSignal != 0 && quality >= effectiveMinQuality)
         {
            if((obSignal > 0 && regime > 0) || (obSignal < 0 && regime < 0))
            {
               signal = obSignal;
               signalType = "ORDER_BLOCK";
            }
         }
      }
   }

   if(signal == 0) return;

   // Entry Trigger Check
   string entryType = "";
   if(!CheckEntryTrigger(signal, entryType))
   {
      if(DebugMode) Print("DEBUG: Skipped - No entry trigger");
      return;
   }

   // Store entry type for risk calculation
   currentEntryType = entryType;

   // LAYER 4: Pattern Filter
   double patternSizeMult = 1.0;
   int patternExtraQ = 0;
   string direction = (signal > 0) ? "BUY" : "SELL";

   if(!CheckPatternFilter(direction, patternSizeMult, patternExtraQ)) return;

   if(patternExtraQ > 0 && quality < effectiveMinQuality + patternExtraQ) return;

   // Execute Trade with v3 risk calculation
   double monthRiskMult = GetMonthlyRiskMult(dt.mon);
   double techMult = (technicalQuality == BASE_QUALITY_GOOD) ? 1.0 : 0.8;

   // v3: Include entry multiplier and apply clamp
   double entryMult = GetEntryMultiplier(entryType);
   double riskMult = dayMult * hourMult * entryMult * (quality / 100.0) * monthRiskMult * patternSizeMult * techMult;

   // v3 FIX: Clamp risk multiplier to [0.30, 1.20] like Python
   if(riskMult < 0.30)
   {
      if(DebugMode) Print("DEBUG: Skipped - Risk mult too low (", riskMult, ")");
      return;
   }
   riskMult = MathMax(0.30, MathMin(1.20, riskMult));

   ExecuteTrade(signal, atr[1], riskMult, signalType + "_" + entryType);
}

//+------------------------------------------------------------------+
//| Check for Order Block signal                                      |
//+------------------------------------------------------------------+
int CheckOrderBlock(double &quality)
{
   double currentClose = iClose(_Symbol, PERIOD_H1, 1);
   int bestSignal = 0;
   double bestQuality = 0;

   double currentBarHigh = iHigh(_Symbol, PERIOD_H1, 1);
   double currentBarLow = iLow(_Symbol, PERIOD_H1, 1);
   double currentZoneSize = (currentBarHigh - currentBarLow) * 2;

   for(int i = 30; i >= 2; i--)
   {
      double open_curr = iOpen(_Symbol, PERIOD_H1, i);
      double close_curr = iClose(_Symbol, PERIOD_H1, i);
      double high_curr = iHigh(_Symbol, PERIOD_H1, i);
      double low_curr = iLow(_Symbol, PERIOD_H1, i);

      double open_next = iOpen(_Symbol, PERIOD_H1, i-1);
      double close_next = iClose(_Symbol, PERIOD_H1, i-1);
      double high_next = iHigh(_Symbol, PERIOD_H1, i-1);
      double low_next = iLow(_Symbol, PERIOD_H1, i-1);

      double range_next = high_next - low_next;
      if(range_next < 0.0003) continue;

      double body_next = MathAbs(close_next - open_next);
      double bodyRatio = body_next / range_next;

      // Bullish OB
      if(close_curr < open_curr)
      {
         if(close_next > open_next && bodyRatio > 0.55 && close_next > high_curr)
         {
            double obQuality = bodyRatio * 100;
            double obPrice = low_curr;
            double zoneSize = currentZoneSize;
            if(MathAbs(currentClose - obPrice) <= zoneSize)
            {
               if(obQuality > bestQuality)
               {
                  bestSignal = 1;
                  bestQuality = obQuality;
               }
            }
         }
      }

      // Bearish OB
      if(close_curr > open_curr)
      {
         if(close_next < open_next && bodyRatio > 0.55 && close_next < low_curr)
         {
            double obQuality = bodyRatio * 100;
            double obPrice = high_curr;
            double zoneSize = currentZoneSize;
            if(MathAbs(currentClose - obPrice) <= zoneSize)
            {
               if(obQuality > bestQuality)
               {
                  bestSignal = -1;
                  bestQuality = obQuality;
               }
            }
         }
      }
   }

   quality = bestQuality;
   return bestSignal;
}

//+------------------------------------------------------------------+
//| Check for EMA Pullback signal                                     |
//+------------------------------------------------------------------+
int CheckEmaPullback(double &emaFast[], double &emaSlow[], double &rsi[],
                     double &adx[], double atrPips, double &quality)
{
   double open1 = iOpen(_Symbol, PERIOD_H1, 1);
   double close1 = iClose(_Symbol, PERIOD_H1, 1);
   double high1 = iHigh(_Symbol, PERIOD_H1, 1);
   double low1 = iLow(_Symbol, PERIOD_H1, 1);

   double range1 = high1 - low1;
   if(range1 < 0.0003) return 0;

   double body1 = MathAbs(close1 - open1);
   double bodyRatio = body1 / range1;

   if(bodyRatio < 0.4) return 0;
   if(adx[1] < 20) return 0;
   if(rsi[1] < 30 || rsi[1] > 70) return 0;

   double atrDistance = atrPips * pipSize * 1.5;

   // BUY
   bool isBullish = (close1 > open1);
   if(isBullish && close1 > emaFast[1] && emaFast[1] > emaSlow[1])
   {
      double distanceToEma = low1 - emaFast[1];
      if(distanceToEma <= atrDistance)
      {
         double touchQ = MathMax(0, 30 - (distanceToEma / pipSize));
         double adxQ = MathMin(25, (adx[1] - 15) * 1.5);
         double rsiQ = (MathAbs(50 - rsi[1]) < 20) ? 25 : 15;
         double bodyQ = MathMin(20, bodyRatio * 30);

         quality = MathMin(100, MathMax(55, touchQ + adxQ + rsiQ + bodyQ));
         return 1;
      }
   }

   // SELL
   bool isBearish = (close1 < open1);
   if(isBearish && close1 < emaFast[1] && emaFast[1] < emaSlow[1])
   {
      double distanceToEma = emaFast[1] - high1;
      if(distanceToEma <= atrDistance)
      {
         double touchQ = MathMax(0, 30 - (distanceToEma / pipSize));
         double adxQ = MathMin(25, (adx[1] - 15) * 1.5);
         double rsiQ = (MathAbs(50 - rsi[1]) < 20) ? 25 : 15;
         double bodyQ = MathMin(20, bodyRatio * 30);

         quality = MathMin(100, MathMax(55, touchQ + adxQ + rsiQ + bodyQ));
         return -1;
      }
   }

   return 0;
}

//+------------------------------------------------------------------+
//| Execute trade (v3: NO pre-trade lot cap, uses POST-trade cap)     |
//+------------------------------------------------------------------+
void ExecuteTrade(int signal, double atr, double riskMult, string signalType)
{
   double balance = accInfo.Balance();
   double riskAmount = balance * (RiskPercent / 100.0) * riskMult;

   double slPips = atr / pipSize * SL_ATR_Mult;
   double tpPips = slPips * TP_Ratio;

   // v3 FIX: Calculate lot size WITHOUT pre-trade cap (like Python)
   // Python: lot_size = risk_amount / (sl_pips * PIP_VALUE)
   double lotSize = riskAmount / (slPips * pipValue);
   lotSize = MathMax(MinLotSize, MathMin(MaxLotSize, lotSize));
   lotSize = NormalizeLotSize(lotSize);

   // v3: NO SL_CAPPED here - cap will be applied AFTER trade closes (like Python)
   // This is tracked in OnTradeTransaction

   double price, sl, tp;

   if(signal > 0)
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl = price - slPips * pipSize;
      tp = price + tpPips * pipSize;

      if(trade.Buy(lotSize, _Symbol, price, sl, tp, "QuadV3_" + signalType))
      {
         Print("BUY: ", signalType, " | Lot: ", lotSize, " | SL: ", DoubleToString(slPips, 1), " pips | TP: ", DoubleToString(tpPips, 1), " pips");
      }
   }
   else if(signal < 0)
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl = price + slPips * pipSize;
      tp = price - tpPips * pipSize;

      if(trade.Sell(lotSize, _Symbol, price, sl, tp, "QuadV3_" + signalType))
      {
         Print("SELL: ", signalType, " | Lot: ", lotSize, " | SL: ", DoubleToString(slPips, 1), " pips | TP: ", DoubleToString(tpPips, 1), " pips");
      }
   }
}

//+------------------------------------------------------------------+
//| Normalize lot size                                                |
//+------------------------------------------------------------------+
double NormalizeLotSize(double lots)
{
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   lots = MathMax(MinLotSize, MathMin(MaxLotSize, lots));
   lots = MathMax(minLot, MathMin(maxLot, lots));
   lots = MathFloor(lots / lotStep) * lotStep;

   return NormalizeDouble(lots, 2);
}

//+------------------------------------------------------------------+
//| Check if we have an open position                                 |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(posInfo.SelectByIndex(i))
      {
         if(posInfo.Symbol() == _Symbol && posInfo.Magic() == MagicNumber)
            return true;
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| OnTradeTransaction - v3: Apply POST-trade cap like Python         |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;

   ulong dealTicket = trans.deal;
   if(dealTicket == 0) return;

   if(!HistoryDealSelect(dealTicket)) return;

   long dealMagic = HistoryDealGetInteger(dealTicket, DEAL_MAGIC);
   if(dealMagic != MagicNumber) return;

   ENUM_DEAL_ENTRY dealEntry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(dealTicket, DEAL_ENTRY);
   if(dealEntry != DEAL_ENTRY_OUT) return;

   double dealProfit = HistoryDealGetDouble(dealTicket, DEAL_PROFIT);
   double dealSwap = HistoryDealGetDouble(dealTicket, DEAL_SWAP);
   double dealCommission = HistoryDealGetDouble(dealTicket, DEAL_COMMISSION);
   double totalPnL = dealProfit + dealSwap + dealCommission;

   // v3 FIX: Apply POST-trade cap like Python
   // Python: if total_pnl < 0 and abs(total_pnl) > max_loss: total_pnl = -max_loss
   double balance = accInfo.Balance();
   double maxLoss = balance * (MaxLossPercent / 100.0);

   string cappedNote = "";
   if(totalPnL < 0 && MathAbs(totalPnL) > maxLoss)
   {
      // Note: In live trading, we can't actually change the PnL
      // This is for tracking purposes - the actual cap would need to be
      // implemented via position management (partial close at max loss)
      cappedNote = " [WOULD_CAP to -$" + DoubleToString(maxLoss, 2) + "]";
      // For backtest tracking, we use actual PnL but note the cap
   }

   ENUM_DEAL_TYPE dealType = (ENUM_DEAL_TYPE)HistoryDealGetInteger(dealTicket, DEAL_TYPE);
   string direction = "";
   if(dealType == DEAL_TYPE_SELL) direction = "BUY";
   else if(dealType == DEAL_TYPE_BUY) direction = "SELL";

   // Record for Layer 3 & 4
   RecordTradeForRiskManager(totalPnL);
   RecordTradeForPatternFilter(direction, totalPnL, TimeCurrent());

   string exitType = (totalPnL >= 0) ? "TP" : "SL";
   Print("Trade closed: ", direction, " | P&L: $", DoubleToString(totalPnL, 2), cappedNote,
         " | Monthly: $", DoubleToString(monthlyPnL, 2),
         " | Consec Losses: ", consecutiveLosses);
}

//+------------------------------------------------------------------+
