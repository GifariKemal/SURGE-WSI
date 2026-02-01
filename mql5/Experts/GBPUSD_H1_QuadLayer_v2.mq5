//+------------------------------------------------------------------+
//|                                     GBPUSD_H1_QuadLayer_v2.mq5   |
//|                                    SURGE-WSI Trading System      |
//|                                    Version 2.0 - Backtest Fix    |
//+------------------------------------------------------------------+
#property copyright "SURGE-WSI"
#property link      "https://github.com/surge-wsi"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>

//+------------------------------------------------------------------+
//| Input Parameters                                                  |
//+------------------------------------------------------------------+
input group "=== Risk Management ==="
input double   RiskPercent = 1.0;           // Risk per trade (%)
input double   MaxLossPercent = 0.15;       // Max Loss per trade (%) - SL_CAPPED
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
input bool     BacktestUTC = true;          // [BACKTEST] Data is already UTC (no offset needed)
input bool     AutoDetectGMT = true;        // [LIVE] Auto-detect Broker GMT Offset
input int      GMTOffset = 0;               // [LIVE] Manual GMT Offset (if AutoDetect=false)
input int      LondonStart = 8;             // London Session Start (UTC)
input int      LondonEnd = 10;              // London Session End (UTC)
input int      NewYorkStart = 13;           // New York Session Start (UTC)
input int      NewYorkEnd = 17;             // New York Session End

input group "=== Magic Number ==="
input int      MagicNumber = 69002;         // EA Magic Number (v2)

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

double         pipSize;
double         pipValue;
int            detectedGMTOffset;  // Actual GMT offset to use

// Day Multipliers (v6.9)
// MQL5: Index 0=Sun, 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat
// Python: Mon=1.0, Tue=0.9, Wed=1.0, Thu=0.8, Fri=0.3, Sat/Sun=0.0
double DayMultipliers[7] = {0.0, 1.0, 0.9, 1.0, 0.8, 0.3, 0.0};

// Hour Multipliers
double HourMultipliers[24] = {
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // 0-5
   0.5, 0.0, 1.0, 1.0, 0.9, 0.0,  // 6-11 (7,11 = skip)
   0.7, 1.0, 1.0, 1.0, 0.9, 0.7,  // 12-17
   0.3, 0.0, 0.0, 0.0, 0.0, 0.0   // 18-23
};

// Layer 1: Monthly Tradeable Percentage (from market analysis)
// Higher = better conditions, lower = increase quality requirement
double MonthlyTradeablePct[12] = {
   65,  // Jan - OK
   55,  // Feb - POOR! (needs +15 quality)
   70,  // Mar - Good
   70,  // Apr - Pattern filter handles choppy
   62,  // May - Below avg
   68,  // Jun - OK
   78,  // Jul - Good
   65,  // Aug - Average
   72,  // Sep - Good
   58,  // Oct - Below avg
   66,  // Nov - OK
   60   // Dec - Low (holidays)
};

// Layer 1: Monthly Risk Multipliers (lot size adjustment)
double MonthlyRiskMult[12] = {
   0.9,   // Jan
   0.6,   // Feb - lowered due to poor conditions
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

// Layer 2: Technical Market Condition Thresholds (from Python)
#define ATR_STABILITY_THRESHOLD    0.25   // ATR coefficient of variation
#define EFFICIENCY_THRESHOLD       0.08   // Price movement efficiency
#define TREND_STRENGTH_THRESHOLD   25.0   // ADX threshold
#define BASE_QUALITY_GOOD          60     // Good market conditions
#define BASE_QUALITY_NORMAL        65     // Normal conditions
#define BASE_QUALITY_BAD           80     // Bad conditions - need higher signal quality

// Layer 3: Intra-Month Risk Manager State
int    currentMonth = 0;
int    currentDay = 0;
double monthlyPnL = 0.0;
int    consecutiveLosses = 0;
bool   monthStopped = false;
bool   dayStopped = false;

// Layer 4: Pattern Filter State
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

//+------------------------------------------------------------------+
//| Check if running in Strategy Tester                               |
//+------------------------------------------------------------------+
bool IsBacktest()
{
   return MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_OPTIMIZATION);
}

//+------------------------------------------------------------------+
//| LAYER 1: Get Monthly Quality Adjustment                           |
//| Based on historical tradeable percentage                          |
//+------------------------------------------------------------------+
int GetMonthlyQualityAdjustment(int month)
{
   // month is 1-12, array is 0-11
   double tradeable = MonthlyTradeablePct[month - 1];

   if(tradeable < 30)
      return 50;  // NO TRADE
   else if(tradeable < 40)
      return 35;  // HALT
   else if(tradeable < 50)
      return 25;  // Extreme
   else if(tradeable < 60)
      return 15;  // Very poor (Feb, Oct)
   else if(tradeable < 70)
      return 10;  // Below average
   else if(tradeable < 75)
      return 5;   // Slight
   else
      return 0;   // Good month
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
//| Returns: 0=can trade, 1=month stopped, 2=day stopped              |
//+------------------------------------------------------------------+
int CheckIntraMonthRisk(datetime currentTime, int &dynamicAdj)
{
   MqlDateTime dt;
   TimeToStruct(currentTime, dt);

   int monthKey = dt.year * 100 + dt.mon;
   int dayKey = dt.year * 10000 + dt.mon * 100 + dt.day;

   // Reset for new month
   if(monthKey != currentMonth)
   {
      currentMonth = monthKey;
      monthlyPnL = 0.0;
      consecutiveLosses = 0;
      monthStopped = false;
      dayStopped = false;
   }

   // Reset for new day
   if(dayKey != currentDay)
   {
      currentDay = dayKey;
      dayStopped = false;
   }

   // Check circuit breakers
   if(monthStopped)
      return 1;
   if(dayStopped)
      return 2;

   // Calculate dynamic adjustment
   dynamicAdj = 0;

   // Monthly loss check
   if(monthlyPnL <= MonthlyLossStop)
   {
      monthStopped = true;
      Print("LAYER3: Month stopped at P&L $", monthlyPnL);
      return 1;
   }

   // Consecutive loss check
   if(consecutiveLosses >= ConsecLossMax)
   {
      dayStopped = true;
      Print("LAYER3: Day stopped after ", consecutiveLosses, " consecutive losses");
      return 2;
   }

   // Dynamic quality adjustments based on monthly P&L
   if(monthlyPnL <= -350)
      dynamicAdj = 15;
   else if(monthlyPnL <= -250)
      dynamicAdj = 10;
   else if(monthlyPnL <= -150)
      dynamicAdj = 5;

   // Extra adjustment for consecutive losses
   if(consecutiveLosses >= 3)
      dynamicAdj += 5;

   return 0;  // Can trade
}

//+------------------------------------------------------------------+
//| LAYER 3: Record Trade for Intra-Month Tracking                    |
//+------------------------------------------------------------------+
void RecordTradeForRiskManager(double pnl)
{
   monthlyPnL += pnl;

   if(pnl < 0)
      consecutiveLosses++;
   else
      consecutiveLosses = 0;  // Reset on win

   if(DebugMode)
      Print("LAYER3: Monthly P&L = $", monthlyPnL, ", Consecutive Losses = ", consecutiveLosses);
}

//+------------------------------------------------------------------+
//| LAYER 4: Get Rolling Win Rate                                     |
//+------------------------------------------------------------------+
double GetRollingWinRate()
{
   if(tradeHistoryCount < 3)
      return 1.0;  // Not enough data

   int lookback = MathMin(RollingWindow, tradeHistoryCount);
   int wins = 0;

   for(int i = tradeHistoryCount - lookback; i < tradeHistoryCount; i++)
   {
      if(tradeHistory[i].pnl > 0)
         wins++;
   }

   return (double)wins / (double)lookback;
}

//+------------------------------------------------------------------+
//| LAYER 4: Check Both Directions Failing                            |
//+------------------------------------------------------------------+
bool AreBothDirectionsFailing()
{
   if(tradeHistoryCount < 8)
      return false;

   int lookback = MathMin(8, tradeHistoryCount);
   int buyLosses = 0, sellLosses = 0;

   for(int i = tradeHistoryCount - lookback; i < tradeHistoryCount; i++)
   {
      if(tradeHistory[i].pnl < 0)
      {
         if(tradeHistory[i].direction == "BUY")
            buyLosses++;
         else
            sellLosses++;
      }
   }

   return (buyLosses >= 4 && sellLosses >= 4);
}

//+------------------------------------------------------------------+
//| LAYER 4: Check Pattern Filter                                     |
//| Returns: true=can trade, false=halted                             |
//+------------------------------------------------------------------+
bool CheckPatternFilter(string direction, double &sizeMult, int &extraQuality)
{
   sizeMult = 1.0;
   extraQuality = 0;

   if(!UsePatternFilter)
      return true;

   // During warmup, always allow
   if(tradeHistoryCount < WarmupTrades)
      return true;

   // Check if halted
   if(patternHalted && !inRecovery)
      return false;

   // Check both directions failing
   if(AreBothDirectionsFailing())
   {
      patternHalted = true;
      inRecovery = true;
      recoveryWins = 0;
      Print("LAYER4: HALT - Both directions failing");
      return false;
   }

   // Check rolling win rate
   double rollingWR = GetRollingWinRate();
   if(rollingWR < RollingWRHalt)
   {
      patternHalted = true;
      inRecovery = true;
      recoveryWins = 0;
      Print("LAYER4: HALT - Rolling WR too low: ", rollingWR * 100, "%");
      return false;
   }

   // Recovery mode adjustments
   if(inRecovery)
   {
      sizeMult = 0.5;      // Trade at 50% size
      extraQuality = 5;    // Extra quality requirement
   }
   // Caution mode
   else if(rollingWR < 0.25)
   {
      sizeMult = 0.6;      // Trade at 60% size
      extraQuality = 3;
   }

   return true;
}

//+------------------------------------------------------------------+
//| LAYER 4: Record Trade for Pattern Filter                          |
//+------------------------------------------------------------------+
void RecordTradeForPatternFilter(string direction, double pnl, datetime time)
{
   if(!UsePatternFilter)
      return;

   // Add to history
   if(tradeHistoryCount < 50)
   {
      tradeHistory[tradeHistoryCount].direction = direction;
      tradeHistory[tradeHistoryCount].pnl = pnl;
      tradeHistory[tradeHistoryCount].time = time;
      tradeHistoryCount++;
   }
   else
   {
      // Shift history
      for(int i = 0; i < 49; i++)
         tradeHistory[i] = tradeHistory[i+1];
      tradeHistory[49].direction = direction;
      tradeHistory[49].pnl = pnl;
      tradeHistory[49].time = time;
   }

   // Track recovery
   if(inRecovery)
   {
      if(pnl > 0)
      {
         recoveryWins++;
         if(recoveryWins >= 1)  // Need 1 win to exit recovery
         {
            patternHalted = false;
            inRecovery = false;
            recoveryWins = 0;
            Print("LAYER4: Recovery complete - resuming normal trading");
         }
      }
      else
      {
         recoveryWins = 0;  // Reset on loss
      }
   }
}

//+------------------------------------------------------------------+
//| Check Entry Trigger (MOMENTUM, ENGULF, LOWER_HIGH)                |
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
   if(range1 < 0.0003)
      return false;

   double body1 = MathAbs(close1 - open1);
   double prevBody = MathAbs(close2 - open2);
   bool isBullish = close1 > open1;
   bool isBearish = close1 < open1;
   bool prevBullish = close2 > open2;
   bool prevBearish = close2 < open2;

   // MOMENTUM: Strong candle with body > 50% of range
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

   // ENGULF: Current body > previous body * 1.2 with reversal
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

   // LOWER_HIGH: For SELL - current high lower than previous high
   if(direction < 0 && high1 < high2 && isBearish)
   {
      entryType = "LOWER_HIGH";
      return true;
   }

   return false;
}

//+------------------------------------------------------------------+
//| LAYER 2: Assess Technical Market Condition                        |
//| Returns: technical quality baseline (60-80)                       |
//| Also returns market label for debugging                           |
//+------------------------------------------------------------------+
int AssessTechnicalCondition(string &marketLabel)
{
   int lookback = 20;

   // 1. Calculate ATR Stability (coefficient of variation)
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

   // 2. Calculate Price Efficiency (net move / total move)
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

   // 3. Get Trend Strength from ADX
   double adxValues[];
   ArraySetAsSeries(adxValues, true);
   if(CopyBuffer(adxHandle, 0, 1, 3, adxValues) < 3)
   {
      marketLabel = "NO_DATA";
      return BASE_QUALITY_NORMAL;
   }
   double trendStrength = adxValues[0];

   // 4. Calculate technical score (0-100)
   int score = 0;

   // ATR Stability: Lower CV = more stable = better
   if(atrCV < ATR_STABILITY_THRESHOLD)
      score += 33;
   else if(atrCV < ATR_STABILITY_THRESHOLD * 1.5)
      score += 20;

   // Price Efficiency: Higher = more trending = better
   if(efficiency > EFFICIENCY_THRESHOLD)
      score += 33;
   else if(efficiency > EFFICIENCY_THRESHOLD * 0.5)
      score += 20;

   // Trend Strength: Higher ADX = stronger trend = better
   if(trendStrength > TREND_STRENGTH_THRESHOLD)
      score += 34;
   else if(trendStrength > TREND_STRENGTH_THRESHOLD * 0.7)
      score += 20;

   // 5. Determine technical quality baseline
   int technicalQuality;
   if(score >= 80)
   {
      technicalQuality = BASE_QUALITY_GOOD;  // 60 - Good conditions
      marketLabel = "GOOD";
   }
   else if(score >= 40)
   {
      technicalQuality = BASE_QUALITY_NORMAL;  // 65 - Normal conditions
      marketLabel = "NORMAL";
   }
   else
   {
      technicalQuality = BASE_QUALITY_BAD;  // 80 - Bad conditions, need high quality
      marketLabel = "BAD";
   }

   if(DebugMode)
      Print("LAYER2: ATR_CV=", DoubleToString(atrCV, 3),
            " Efficiency=", DoubleToString(efficiency, 3),
            " ADX=", DoubleToString(trendStrength, 1),
            " Score=", score, " -> ", marketLabel, " (Q=", technicalQuality, ")");

   return technicalQuality;
}

//+------------------------------------------------------------------+
//| Detect broker GMT offset (LIVE TRADING ONLY)                      |
//+------------------------------------------------------------------+
int DetectGMTOffset()
{
   // Live trading: TimeGMT() should work correctly
   datetime gmtTime = TimeGMT();
   datetime serverTime = TimeCurrent();

   // Calculate difference in hours
   int diffSeconds = (int)(serverTime - gmtTime);
   int diffHours = diffSeconds / 3600;

   // Round to nearest hour (handle small differences)
   if(diffSeconds % 3600 > 1800)
      diffHours += 1;
   else if(diffSeconds % 3600 < -1800)
      diffHours -= 1;

   return diffHours;
}

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
   // Set trade parameters
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_IOC);

   // =============================================================
   // GMT OFFSET LOGIC (v2 FIX)
   // =============================================================
   // KEY INSIGHT: MT5 Strategy Tester data is ALWAYS in UTC!
   // - When backtesting, the bar timestamps are UTC, not broker time
   // - So we should NOT apply any GMT offset to convert to UTC
   // - The data is already UTC!
   //
   // For LIVE trading:
   // - TimeCurrent() returns broker server time
   // - We need to convert to UTC using the broker's GMT offset
   // =============================================================

   if(IsBacktest())
   {
      // BACKTEST MODE
      if(BacktestUTC)
      {
         detectedGMTOffset = 0;  // Data is already UTC, no conversion needed
         Print("=== BACKTEST MODE (v2) ===");
         Print("BacktestUTC=true: Data is already UTC, GMTOffset set to 0");
         Print("Kill zones will be correctly identified at UTC hours");
      }
      else
      {
         // User wants to apply manual offset (legacy mode)
         detectedGMTOffset = GMTOffset;
         Print("=== BACKTEST MODE (LEGACY) ===");
         Print("BacktestUTC=false: Using manual GMTOffset=", GMTOffset);
         Print("WARNING: This may cause incorrect kill zone detection!");
      }
   }
   else
   {
      // LIVE TRADING MODE
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

   // Log important time info
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int utcHour = dt.hour - detectedGMTOffset;
   if(utcHour < 0) utcHour += 24;
   if(utcHour >= 24) utcHour -= 24;

   Print("Current Time: ", TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES),
         " | Effective UTC Hour: ", utcHour,
         " | Day: ", EnumToString((ENUM_DAY_OF_WEEK)dt.day_of_week));

   // Calculate pip size and value
   pipSize = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(StringFind(_Symbol, "JPY") >= 0)
      pipSize *= 100;
   else
      pipSize *= 10;

   pipValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE) * 10;

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

   Print("=== GBPUSD H1 QuadLayer v2.0 (Backtest Fix) initialized ===");
   Print("Risk: ", RiskPercent, "% | Max Loss (SL_CAPPED): ", MaxLossPercent, "% | SL: ", SL_ATR_Mult, "x ATR | TP: ", TP_Ratio, ":1");
   Print("Trading Hours (UTC): London ", LondonStart, "-", LondonEnd, " | NY ", NewYorkStart, "-", NewYorkEnd);
   Print("Day Multipliers: Mon=1.0, Tue=0.9, Wed=1.0, Thu=0.8, Fri=0.3");
   Print("--- QUAD-LAYER QUALITY FILTER ---");
   Print("Layer 1: Monthly Profile (quality adj by month)");
   Print("Layer 2: Technical (ATR stability, efficiency, trend)");
   Print("Layer 3: Intra-Month Risk (stop at $", MonthlyLossStop, ", ", ConsecLossMax, " consec losses)");
   Print("Layer 4: Pattern Filter (halt at ", RollingWRHalt*100, "% WR, warmup=", WarmupTrades, ")");
   Print("Entry Trigger: ", UseEntryTrigger ? "ENABLED (MOMENTUM/ENGULF/LOWER_HIGH)" : "DISABLED");

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

   Print("GBPUSD H1 QuadLayer v2.0 deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // Only trade on new bar
   static datetime lastBar = 0;
   static datetime firstBar = 0;  // Track first bar of backtest
   static int barCount = 0;       // Count bars since start

   datetime currentBar = iTime(_Symbol, PERIOD_H1, 0);
   if(lastBar == currentBar) return;
   lastBar = currentBar;

   // Initialize first bar tracking
   if(firstBar == 0)
   {
      firstBar = currentBar;
      barCount = 0;
   }
   barCount++;

   // SYNC FIX: 100-bar warmup period to match Python backtest
   // Python starts at bar 100, so we skip first 100 bars
   if(barCount < 100)
   {
      if(DebugMode) Print("DEBUG: Warmup - bar ", barCount, "/100");
      return;
   }

   // Check if we already have a position
   if(HasOpenPosition()) return;

   // Get current time info and convert to UTC
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);

   // Convert server hour to UTC hour using detected/configured offset
   int utcHour = dt.hour - detectedGMTOffset;
   if(utcHour < 0) utcHour += 24;
   if(utcHour >= 24) utcHour -= 24;

   // Check day filter (skip weekends)
   if(dt.day_of_week == 0 || dt.day_of_week == 6)
   {
      if(DebugMode) Print("DEBUG: Skipped - Weekend (Day=", dt.day_of_week, ")");
      return;
   }

   // Check day multiplier
   double dayMult = DayMultipliers[dt.day_of_week];
   if(dayMult <= 0.0)
   {
      if(DebugMode) Print("DEBUG: Skipped - DayMult=0 (Day=", dt.day_of_week, ")");
      return;
   }

   // Check hour multiplier (using UTC hour)
   double hourMult = HourMultipliers[utcHour];
   if(hourMult <= 0.0)
   {
      if(DebugMode) Print("DEBUG: Skipped - HourMult=0 (UTC Hour=", utcHour, ", Server Hour=", dt.hour, ")");
      return;
   }

   // Check if in kill zone (using UTC hour)
   bool inLondon = (utcHour >= LondonStart && utcHour <= LondonEnd);
   bool inNewYork = (utcHour >= NewYorkStart && utcHour <= NewYorkEnd);
   if(!inLondon && !inNewYork)
   {
      if(DebugMode) Print("DEBUG: Skipped - Outside Kill Zone (UTC Hour=", utcHour, ", London=", LondonStart, "-", LondonEnd, ", NY=", NewYorkStart, "-", NewYorkEnd, ")");
      return;
   }

   // Skip Hour 11 if enabled (using UTC hour)
   if(SkipHour11 && utcHour == 11)
   {
      if(DebugMode) Print("DEBUG: Skipped - Hour 11 Filter");
      return;
   }

   // ============================================================
   // LAYER 3: Check Intra-Month Risk Manager
   // ============================================================
   int dynamicAdj = 0;
   int riskCheck = CheckIntraMonthRisk(TimeCurrent(), dynamicAdj);
   if(riskCheck == 1)
   {
      if(DebugMode) Print("DEBUG: Skipped - Month stopped (Layer 3)");
      return;
   }
   if(riskCheck == 2)
   {
      if(DebugMode) Print("DEBUG: Skipped - Day stopped (Layer 3)");
      return;
   }

   // Get indicator values
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

   // Convert ATR to pips
   // pipSize = 0.0001 for GBPUSD (1 pip), so ATR/pipSize = pips directly
   double atrPips = atr[1] / pipSize;

   // Check ATR range
   if(atrPips < MinATR_Pips || atrPips > MaxATR_Pips)
   {
      if(DebugMode) Print("DEBUG: Skipped - ATR out of range (ATR=", DoubleToString(atrPips, 1), ", Min=", MinATR_Pips, ", Max=", MaxATR_Pips, ")");
      return;
   }

   // Detect regime
   double close = iClose(_Symbol, PERIOD_H1, 1);
   int regime = 0; // 0=sideways, 1=bullish, -1=bearish

   if(close > emaFast[1] && emaFast[1] > emaSlow[1])
      regime = 1;  // Bullish
   else if(close < emaFast[1] && emaFast[1] < emaSlow[1])
      regime = -1; // Bearish

   if(regime == 0)
   {
      if(DebugMode) Print("DEBUG: Skipped - Sideways Regime (Close=", close, ", EMA20=", emaFast[1], ", EMA50=", emaSlow[1], ")");
      return;
   }

   // ============================================================
   // LAYER 2: Technical Market Condition Assessment
   // ============================================================
   string marketLabel = "";
   int technicalQuality = AssessTechnicalCondition(marketLabel);

   // SYNC FIX v6.99: Python allows BAD market trades with higher quality threshold
   // Previously MQL5 completely rejected BAD conditions, but Python continues
   // with technicalQuality = 80 (requiring higher quality signals)
   // REMOVED: Early return for BAD conditions
   if(DebugMode && technicalQuality >= BASE_QUALITY_BAD)
      Print("DEBUG: BAD conditions - using higher quality threshold (", technicalQuality, ")");

   // ============================================================
   // LAYER 1: Monthly Quality Adjustment
   // ============================================================
   int monthlyAdj = GetMonthlyQualityAdjustment(dt.mon);

   // Combined quality threshold: Technical base + Monthly adj + Dynamic adj
   double effectiveMinQuality = technicalQuality + monthlyAdj + dynamicAdj;

   if(DebugMode && (monthlyAdj > 0 || dynamicAdj > 0))
      Print("DEBUG: Quality adjusted: tech=", technicalQuality, " + monthly=", monthlyAdj, " + dynamic=", dynamicAdj, " = ", effectiveMinQuality);

   // Check for entry signals
   int signal = 0;
   string signalType = "";
   double quality = 0;

   // ============================================================
   // SIGNAL PRIORITY: EMA Pullback FIRST (74% of trades in Python)
   // ============================================================

   // Entry Signal 1: EMA Pullback (higher priority - better quality)
   if(UseEmaPullback)
   {
      // Session filter BEFORE detection (like Python)
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
            else if(DebugMode)
            {
               Print("DEBUG: EMA_PULLBACK rejected - regime mismatch (signal=", emaSignal, ", regime=", regime, ")");
            }
         }
      }
   }

   // Entry Signal 2: Order Block (fallback if no EMA signal)
   if(signal == 0 && UseOrderBlock)
   {
      // Session filter BEFORE detection (like Python)
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
            else if(DebugMode)
            {
               Print("DEBUG: ORDER_BLOCK rejected - regime mismatch (signal=", obSignal, ", regime=", regime, ")");
            }
         }
      }
   }

   // No signal found
   if(signal == 0)
      return;

   // ============================================================
   // Entry Trigger Check (MOMENTUM, ENGULF, LOWER_HIGH)
   // ============================================================
   string entryType = "";
   if(!CheckEntryTrigger(signal, entryType))
   {
      if(DebugMode) Print("DEBUG: Skipped - No entry trigger (", signalType, ")");
      return;
   }

   // ============================================================
   // LAYER 4: Pattern Filter Check
   // ============================================================
   double patternSizeMult = 1.0;
   int patternExtraQ = 0;
   string direction = (signal > 0) ? "BUY" : "SELL";

   if(!CheckPatternFilter(direction, patternSizeMult, patternExtraQ))
   {
      if(DebugMode) Print("DEBUG: Skipped - Pattern filter halt (Layer 4)");
      return;
   }

   // Apply pattern extra quality
   if(patternExtraQ > 0 && quality < effectiveMinQuality + patternExtraQ)
   {
      if(DebugMode) Print("DEBUG: Skipped - Quality below pattern threshold (", quality, " < ", effectiveMinQuality + patternExtraQ, ")");
      return;
   }

   // ============================================================
   // Execute Trade with all layer adjustments
   // ============================================================
   double monthRiskMult = GetMonthlyRiskMult(dt.mon);

   // Technical quality multiplier: reduce risk in normal conditions, full risk in good
   double techMult = (technicalQuality == BASE_QUALITY_GOOD) ? 1.0 : 0.8;

   double riskMult = dayMult * hourMult * (quality / 100.0) * monthRiskMult * patternSizeMult * techMult;
   ExecuteTrade(signal, atr[1], riskMult, signalType + "_" + entryType);
}

//+------------------------------------------------------------------+
//| Check for Order Block signal (SYNC v6.99: scan last 30 bars)     |
//+------------------------------------------------------------------+
int CheckOrderBlock(double &quality)
{
   // SYNC FIX: Python scans last 30 bars for Order Blocks, not just 2
   // This matches: for i in range(len(df) - 30, len(df) - 2)

   // SYNC FIX v6.991: Use bar 1 close for proximity check (matches Python)
   // Python uses current_bar close (signal bar), which is bar 1 in MQL5
   double currentClose = iClose(_Symbol, PERIOD_H1, 1);
   int bestSignal = 0;
   double bestQuality = 0;
   double bestPrice = 0;

   // SYNC FIX: Use current bar (bar 1) range for zone size, like Python does
   double currentBarHigh = iHigh(_Symbol, PERIOD_H1, 1);
   double currentBarLow = iLow(_Symbol, PERIOD_H1, 1);
   double currentZoneSize = (currentBarHigh - currentBarLow) * 2;

   // Scan last 30 bars for Order Block formations
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

      // Bullish Order Block: Current bearish, next bullish engulf
      if(close_curr < open_curr)  // Current bar bearish
      {
         if(close_next > open_next && bodyRatio > 0.55 && close_next > high_curr)
         {
            double obQuality = bodyRatio * 100;
            double obPrice = low_curr;  // Order Block zone = low of bearish bar

            // Check if current price is near the OB zone
            double zoneSize = currentZoneSize;  // Use current bar's range, not OB bar's
            if(MathAbs(currentClose - obPrice) <= zoneSize)
            {
               if(obQuality > bestQuality)
               {
                  bestSignal = 1;  // BUY
                  bestQuality = obQuality;
                  bestPrice = obPrice;
               }
            }
         }
      }

      // Bearish Order Block: Current bullish, next bearish engulf
      if(close_curr > open_curr)  // Current bar bullish
      {
         if(close_next < open_next && bodyRatio > 0.55 && close_next < low_curr)
         {
            double obQuality = bodyRatio * 100;
            double obPrice = high_curr;  // Order Block zone = high of bullish bar

            // Check if current price is near the OB zone
            double zoneSize = currentZoneSize;  // Use current bar's range, not OB bar's
            if(MathAbs(currentClose - obPrice) <= zoneSize)
            {
               if(obQuality > bestQuality)
               {
                  bestSignal = -1;  // SELL
                  bestQuality = obQuality;
                  bestPrice = obPrice;
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

   // Filter criteria
   if(bodyRatio < 0.4) return 0;
   if(adx[1] < 20) return 0;
   if(rsi[1] < 30 || rsi[1] > 70) return 0;

   // ATR distance for pullback
   double atrDistance = atrPips * pipSize * 1.5;

   // BUY: Uptrend pullback
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
         return 1; // BUY
      }
   }

   // SELL: Downtrend pullback
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
         return -1; // SELL
      }
   }

   return 0;
}

//+------------------------------------------------------------------+
//| Execute trade                                                     |
//+------------------------------------------------------------------+
void ExecuteTrade(int signal, double atr, double riskMult, string signalType)
{
   double balance = accInfo.Balance();
   double riskAmount = balance * (RiskPercent / 100.0) * riskMult;

   double slPips = atr / pipSize * SL_ATR_Mult;
   double tpPips = slPips * TP_Ratio;

   // Calculate initial lot size based on risk amount
   double lotSize = riskAmount / (slPips * pipValue);

   // SL_CAPPED: Cap the lot size based on MaxLossPercent (0.15%)
   // This ensures max loss per trade is capped regardless of SL distance
   double maxAllowedLoss = balance * (MaxLossPercent / 100.0);
   double potentialLoss = lotSize * slPips * pipValue;

   if(potentialLoss > maxAllowedLoss)
   {
      // Reduce lot size to cap the max loss
      double cappedLotSize = maxAllowedLoss / (slPips * pipValue);
      Print("SL_CAPPED: Reducing lot from ", DoubleToString(lotSize, 2),
            " to ", DoubleToString(cappedLotSize, 2),
            " | Max Loss: $", DoubleToString(maxAllowedLoss, 2));
      lotSize = cappedLotSize;
   }

   lotSize = NormalizeLotSize(lotSize);

   double price, sl, tp;

   if(signal > 0) // BUY
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl = price - slPips * pipSize;
      tp = price + tpPips * pipSize;

      if(trade.Buy(lotSize, _Symbol, price, sl, tp, "QuadLayer_v2_" + signalType))
      {
         Print("BUY executed: ", signalType, " | Lot: ", lotSize, " | SL: ", slPips, " pips | TP: ", tpPips, " pips");
      }
   }
   else if(signal < 0) // SELL
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl = price + slPips * pipSize;
      tp = price - tpPips * pipSize;

      if(trade.Sell(lotSize, _Symbol, price, sl, tp, "QuadLayer_v2_" + signalType))
      {
         Print("SELL executed: ", signalType, " | Lot: ", lotSize, " | SL: ", slPips, " pips | TP: ", tpPips, " pips");
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
//| OnTradeTransaction - Track closed trades for Layer 3 & 4          |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
   // Only process deal additions (trade closures)
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD)
      return;

   // Get deal info
   ulong dealTicket = trans.deal;
   if(dealTicket == 0)
      return;

   // Select the deal
   if(!HistoryDealSelect(dealTicket))
      return;

   // Check if it's our EA's deal
   long dealMagic = HistoryDealGetInteger(dealTicket, DEAL_MAGIC);
   if(dealMagic != MagicNumber)
      return;

   // Check if it's exit deal (not entry)
   ENUM_DEAL_ENTRY dealEntry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(dealTicket, DEAL_ENTRY);
   if(dealEntry != DEAL_ENTRY_OUT)
      return;

   // Get deal P&L
   double dealProfit = HistoryDealGetDouble(dealTicket, DEAL_PROFIT);
   double dealSwap = HistoryDealGetDouble(dealTicket, DEAL_SWAP);
   double dealCommission = HistoryDealGetDouble(dealTicket, DEAL_COMMISSION);
   double totalPnL = dealProfit + dealSwap + dealCommission;

   // Get direction
   ENUM_DEAL_TYPE dealType = (ENUM_DEAL_TYPE)HistoryDealGetInteger(dealTicket, DEAL_TYPE);
   string direction = "";
   // Exit deal type is opposite of position type
   if(dealType == DEAL_TYPE_SELL)
      direction = "BUY";   // Closing a BUY position
   else if(dealType == DEAL_TYPE_BUY)
      direction = "SELL";  // Closing a SELL position

   // Record for Layer 3: Intra-Month Risk Manager
   RecordTradeForRiskManager(totalPnL);

   // Record for Layer 4: Pattern Filter
   RecordTradeForPatternFilter(direction, totalPnL, TimeCurrent());

   // Log
   string exitType = (totalPnL >= 0) ? "TP" : "SL";
   Print("Trade closed: ", direction, " | P&L: $", DoubleToString(totalPnL, 2),
         " | Monthly: $", DoubleToString(monthlyPnL, 2),
         " | Consec Losses: ", consecutiveLosses);
}

//+------------------------------------------------------------------+
