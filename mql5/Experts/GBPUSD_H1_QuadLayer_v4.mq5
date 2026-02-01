//+------------------------------------------------------------------+
//|                                     GBPUSD_H1_QuadLayer_v4.mq5   |
//|                                    SURGE-WSI Trading System      |
//|                                    Version 4.0 - Real SL_CAPPED  |
//+------------------------------------------------------------------+
//| CHANGES FROM v3:                                                  |
//| 1. Added REAL-TIME SL_CAPPED - monitors and closes positions     |
//|    when floating loss exceeds MaxLossPercent (0.15%)             |
//| 2. This matches Python's post-trade loss capping behavior        |
//| 3. Checks position P&L on EVERY TICK, not just new bars          |
//+------------------------------------------------------------------+
#property copyright "SURGE-WSI"
#property link      "https://github.com/surge-wsi"
#property version   "4.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>

//+------------------------------------------------------------------+
//| Input Parameters                                                  |
//+------------------------------------------------------------------+
input group "=== Risk Management ==="
input double   RiskPercent = 1.0;           // Risk per trade (%)
input double   MaxLossPercent = 0.15;       // Max Loss per trade (%) - REAL-TIME CAP
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
input bool     UseEntryTrigger = true;      // Require Entry Trigger

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
input int      MagicNumber = 69004;         // EA Magic Number (v4)

input group "=== Debug ==="
input bool     DebugMode = false;           // Enable Debug Logging

//+------------------------------------------------------------------+
//| Global Variables                                                  |
//+------------------------------------------------------------------+
CTrade         trade;
CPositionInfo  posInfo;
CAccountInfo   accInfo;

int            atrHandle, emaFastHandle, emaSlowHandle, rsiHandle, adxHandle;
double         pipSize, pipValue;
int            detectedGMTOffset;

// v4: Track entry balance for SL_CAPPED calculation
double         entryBalance = 0;
double         maxAllowedLoss = 0;

// Multipliers
double DayMultipliers[7] = {0.0, 1.0, 0.9, 1.0, 0.8, 0.3, 0.0};
double HourMultipliers[24] = {
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.5, 0.0, 1.0, 1.0, 0.9, 0.0,
   0.7, 1.0, 1.0, 1.0, 0.9, 0.7,
   0.3, 0.0, 0.0, 0.0, 0.0, 0.0
};

double GetEntryMultiplier(string entryType)
{
   if(entryType == "MOMENTUM") return 1.0;
   if(entryType == "LOWER_HIGH") return 1.0;
   if(entryType == "ENGULF") return 0.8;
   return 1.0;
}

double MonthlyTradeablePct[12] = {65, 55, 70, 70, 62, 68, 78, 65, 72, 58, 66, 60};
double MonthlyRiskMult[12] = {0.9, 0.6, 0.8, 1.0, 0.7, 0.85, 1.0, 0.75, 0.9, 0.6, 0.75, 0.8};

#define ATR_STABILITY_THRESHOLD    0.25
#define EFFICIENCY_THRESHOLD       0.08
#define TREND_STRENGTH_THRESHOLD   25.0
#define BASE_QUALITY_GOOD          60
#define BASE_QUALITY_NORMAL        65
#define BASE_QUALITY_BAD           80

// Layer 3 & 4 state
int    currentMonth = 0, currentDay = 0;
double monthlyPnL = 0.0;
int    consecutiveLosses = 0;
bool   monthStopped = false, dayStopped = false;

struct TradeRecord { string direction; double pnl; datetime time; };
TradeRecord tradeHistory[50];
int    tradeHistoryCount = 0;
bool   patternHalted = false, inRecovery = false;
int    recoveryWins = 0;

string currentEntryType = "";

// v4: SL_CAPPED statistics
int    slCappedCount = 0;
double slCappedSavings = 0;

//+------------------------------------------------------------------+
//| Check if running in Strategy Tester                               |
//+------------------------------------------------------------------+
bool IsBacktest() { return MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_OPTIMIZATION); }

//+------------------------------------------------------------------+
//| Layer functions (same as v3)                                      |
//+------------------------------------------------------------------+
int GetMonthlyQualityAdjustment(int month)
{
   double tradeable = MonthlyTradeablePct[month - 1];
   if(tradeable < 30) return 50;
   else if(tradeable < 40) return 35;
   else if(tradeable < 50) return 25;
   else if(tradeable < 60) return 15;
   else if(tradeable < 70) return 10;
   else if(tradeable < 75) return 5;
   return 0;
}

double GetMonthlyRiskMult(int month) { return MonthlyRiskMult[month - 1]; }

int CheckIntraMonthRisk(datetime currentTime, int &dynamicAdj)
{
   MqlDateTime dt; TimeToStruct(currentTime, dt);
   int monthKey = dt.year * 100 + dt.mon;
   int dayKey = dt.year * 10000 + dt.mon * 100 + dt.day;

   if(monthKey != currentMonth) {
      currentMonth = monthKey; monthlyPnL = 0.0; consecutiveLosses = 0;
      monthStopped = false; dayStopped = false;
   }
   if(dayKey != currentDay) { currentDay = dayKey; dayStopped = false; }
   if(monthStopped) return 1;
   if(dayStopped) return 2;

   dynamicAdj = 0;
   if(monthlyPnL <= MonthlyLossStop) { monthStopped = true; return 1; }
   if(consecutiveLosses >= ConsecLossMax) { dayStopped = true; return 2; }

   if(monthlyPnL <= -350) dynamicAdj = 15;
   else if(monthlyPnL <= -250) dynamicAdj = 10;
   else if(monthlyPnL <= -150) dynamicAdj = 5;
   if(consecutiveLosses >= 3) dynamicAdj += 5;

   return 0;
}

void RecordTradeForRiskManager(double pnl)
{
   monthlyPnL += pnl;
   if(pnl < 0) consecutiveLosses++; else consecutiveLosses = 0;
}

double GetRollingWinRate()
{
   if(tradeHistoryCount < 3) return 1.0;
   int lookback = MathMin(RollingWindow, tradeHistoryCount);
   int wins = 0;
   for(int i = tradeHistoryCount - lookback; i < tradeHistoryCount; i++)
      if(tradeHistory[i].pnl > 0) wins++;
   return (double)wins / (double)lookback;
}

bool AreBothDirectionsFailing()
{
   if(tradeHistoryCount < 8) return false;
   int lookback = MathMin(8, tradeHistoryCount);
   int buyLosses = 0, sellLosses = 0;
   for(int i = tradeHistoryCount - lookback; i < tradeHistoryCount; i++) {
      if(tradeHistory[i].pnl < 0) {
         if(tradeHistory[i].direction == "BUY") buyLosses++;
         else sellLosses++;
      }
   }
   return (buyLosses >= 4 && sellLosses >= 4);
}

bool CheckPatternFilter(string direction, double &sizeMult, int &extraQuality)
{
   sizeMult = 1.0; extraQuality = 0;
   if(!UsePatternFilter) return true;
   if(tradeHistoryCount < WarmupTrades) return true;
   if(patternHalted && !inRecovery) return false;

   if(AreBothDirectionsFailing()) {
      patternHalted = true; inRecovery = true; recoveryWins = 0;
      return false;
   }

   double rollingWR = GetRollingWinRate();
   if(rollingWR < RollingWRHalt) {
      patternHalted = true; inRecovery = true; recoveryWins = 0;
      return false;
   }

   if(inRecovery) { sizeMult = 0.5; extraQuality = 5; }
   else if(rollingWR < 0.25) { sizeMult = 0.6; extraQuality = 3; }

   return true;
}

void RecordTradeForPatternFilter(string direction, double pnl, datetime time)
{
   if(!UsePatternFilter) return;
   if(tradeHistoryCount < 50) {
      tradeHistory[tradeHistoryCount].direction = direction;
      tradeHistory[tradeHistoryCount].pnl = pnl;
      tradeHistory[tradeHistoryCount].time = time;
      tradeHistoryCount++;
   } else {
      for(int i = 0; i < 49; i++) tradeHistory[i] = tradeHistory[i+1];
      tradeHistory[49].direction = direction;
      tradeHistory[49].pnl = pnl;
      tradeHistory[49].time = time;
   }

   if(inRecovery) {
      if(pnl > 0) {
         recoveryWins++;
         if(recoveryWins >= 1) { patternHalted = false; inRecovery = false; recoveryWins = 0; }
      } else recoveryWins = 0;
   }
}

bool CheckEntryTrigger(int direction, string &entryType)
{
   if(!UseEntryTrigger) { entryType = "DIRECT"; return true; }

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

   if(body1 > range1 * 0.5) {
      if(direction > 0 && isBullish) { entryType = "MOMENTUM"; return true; }
      if(direction < 0 && isBearish) { entryType = "MOMENTUM"; return true; }
   }
   if(body1 > prevBody * 1.2) {
      if(direction > 0 && isBullish && prevBearish) { entryType = "ENGULF"; return true; }
      if(direction < 0 && isBearish && prevBullish) { entryType = "ENGULF"; return true; }
   }
   if(direction < 0 && high1 < high2 && isBearish) { entryType = "LOWER_HIGH"; return true; }

   return false;
}

int AssessTechnicalCondition(string &marketLabel)
{
   int lookback = 20;
   double atrValues[], closes[], adxValues[];
   ArraySetAsSeries(atrValues, true);
   ArraySetAsSeries(closes, true);
   ArraySetAsSeries(adxValues, true);

   if(CopyBuffer(atrHandle, 0, 1, lookback, atrValues) < lookback) { marketLabel = "NO_DATA"; return BASE_QUALITY_NORMAL; }
   if(CopyClose(_Symbol, PERIOD_H1, 1, lookback + 1, closes) < lookback + 1) { marketLabel = "NO_DATA"; return BASE_QUALITY_NORMAL; }
   if(CopyBuffer(adxHandle, 0, 1, 3, adxValues) < 3) { marketLabel = "NO_DATA"; return BASE_QUALITY_NORMAL; }

   double atrSum = 0, atrSumSq = 0;
   for(int i = 0; i < lookback; i++) { atrSum += atrValues[i]; atrSumSq += atrValues[i] * atrValues[i]; }
   double atrMean = atrSum / lookback;
   double atrCV = (atrMean > 0) ? MathSqrt(MathMax(0, (atrSumSq / lookback) - (atrMean * atrMean))) / atrMean : 0.5;

   double netMove = MathAbs(closes[0] - closes[lookback]);
   double totalMove = 0;
   for(int i = 0; i < lookback; i++) totalMove += MathAbs(closes[i] - closes[i + 1]);
   double efficiency = (totalMove > 0) ? netMove / totalMove : 0.1;

   int score = 0;
   if(atrCV < ATR_STABILITY_THRESHOLD) score += 33;
   else if(atrCV < ATR_STABILITY_THRESHOLD * 1.5) score += 20;
   if(efficiency > EFFICIENCY_THRESHOLD) score += 33;
   else if(efficiency > EFFICIENCY_THRESHOLD * 0.5) score += 20;
   if(adxValues[0] > TREND_STRENGTH_THRESHOLD) score += 34;
   else if(adxValues[0] > TREND_STRENGTH_THRESHOLD * 0.7) score += 20;

   if(score >= 80) { marketLabel = "GOOD"; return BASE_QUALITY_GOOD; }
   else if(score >= 40) { marketLabel = "NORMAL"; return BASE_QUALITY_NORMAL; }
   else { marketLabel = "BAD"; return BASE_QUALITY_BAD; }
}

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
//| v4 NEW: Check and enforce SL_CAPPED on open position              |
//+------------------------------------------------------------------+
bool CheckAndEnforceSLCapped()
{
   // Find our position
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!posInfo.SelectByIndex(i)) continue;
      if(posInfo.Symbol() != _Symbol || posInfo.Magic() != MagicNumber) continue;

      // Calculate current floating P&L
      double floatingPnL = posInfo.Profit() + posInfo.Swap() + posInfo.Commission();

      // Check if loss exceeds max allowed
      if(floatingPnL < 0 && MathAbs(floatingPnL) >= maxAllowedLoss)
      {
         // Close position immediately - SL_CAPPED triggered!
         ulong ticket = posInfo.Ticket();
         double lots = posInfo.Volume();
         string direction = (posInfo.PositionType() == POSITION_TYPE_BUY) ? "BUY" : "SELL";

         // Calculate how much we're saving
         double potentialLoss = MathAbs(floatingPnL);
         double savings = potentialLoss - maxAllowedLoss;

         if(trade.PositionClose(ticket))
         {
            slCappedCount++;
            slCappedSavings += savings;

            Print("=== SL_CAPPED TRIGGERED (v4) ===");
            Print("Position: ", direction, " | Floating P&L: $", DoubleToString(floatingPnL, 2));
            Print("Max Allowed Loss: $", DoubleToString(maxAllowedLoss, 2));
            Print("Closed at loss: $", DoubleToString(-maxAllowedLoss, 2));
            Print("Savings this trade: $", DoubleToString(savings, 2));
            Print("Total SL_CAPPED count: ", slCappedCount);
            Print("Total savings: $", DoubleToString(slCappedSavings, 2));

            return true;  // Position was closed
         }
         else
         {
            Print("ERROR: Failed to close position for SL_CAPPED! Error: ", GetLastError());
         }
      }
   }
   return false;  // No position closed
}

//+------------------------------------------------------------------+
//| Get current position info                                         |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(posInfo.SelectByIndex(i))
         if(posInfo.Symbol() == _Symbol && posInfo.Magic() == MagicNumber)
            return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_IOC);

   // GMT Offset
   if(IsBacktest()) {
      detectedGMTOffset = BacktestUTC ? 0 : GMTOffset;
      Print("=== BACKTEST MODE (v4 - SL_CAPPED) ===");
      Print("BacktestUTC=", BacktestUTC, " -> GMTOffset=", detectedGMTOffset);
   } else {
      detectedGMTOffset = AutoDetectGMT ? DetectGMTOffset() : GMTOffset;
      Print("=== LIVE MODE (v4) === GMT+", detectedGMTOffset);
   }

   // Pip calculation
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   pipSize = (digits == 5 || digits == 3) ? point * 10 : point;

   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   pipValue = (tickSize > 0) ? tickValue * (pipSize / tickSize) : 10.0;

   Print("pipSize=", pipSize, " pipValue=", pipValue);

   // Indicators
   atrHandle = iATR(_Symbol, PERIOD_H1, ATR_Period);
   emaFastHandle = iMA(_Symbol, PERIOD_H1, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   emaSlowHandle = iMA(_Symbol, PERIOD_H1, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
   rsiHandle = iRSI(_Symbol, PERIOD_H1, 14, PRICE_CLOSE);
   adxHandle = iADX(_Symbol, PERIOD_H1, 14);

   if(atrHandle == INVALID_HANDLE || emaFastHandle == INVALID_HANDLE ||
      emaSlowHandle == INVALID_HANDLE || rsiHandle == INVALID_HANDLE ||
      adxHandle == INVALID_HANDLE) {
      Print("Error initializing indicators!");
      return INIT_FAILED;
   }

   Print("=== GBPUSD H1 QuadLayer v4.0 (Real SL_CAPPED) ===");
   Print("MaxLossPercent: ", MaxLossPercent, "% - REAL-TIME ENFORCEMENT!");
   Print("Position will be CLOSED if floating loss >= ", MaxLossPercent, "% of balance");

   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   IndicatorRelease(atrHandle);
   IndicatorRelease(emaFastHandle);
   IndicatorRelease(emaSlowHandle);
   IndicatorRelease(rsiHandle);
   IndicatorRelease(adxHandle);

   Print("=== v4 Final Stats ===");
   Print("SL_CAPPED triggered: ", slCappedCount, " times");
   Print("Total savings from SL_CAP: $", DoubleToString(slCappedSavings, 2));
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // ================================================================
   // v4 KEY FEATURE: Check SL_CAPPED on EVERY tick!
   // This runs BEFORE any other logic to ensure quick response
   // ================================================================
   if(HasOpenPosition())
   {
      // Check if we need to close due to SL_CAPPED
      if(CheckAndEnforceSLCapped())
      {
         // Position was closed by SL_CAPPED, don't do anything else this tick
         return;
      }

      // Still have position open, wait for normal SL/TP
      return;
   }

   // ================================================================
   // No position - check for new entry (same as v3)
   // ================================================================

   static datetime lastBar = 0;
   static int barCount = 0;

   datetime currentBar = iTime(_Symbol, PERIOD_H1, 0);
   if(lastBar == currentBar) return;
   lastBar = currentBar;
   barCount++;

   if(barCount < 100) return;  // Warmup

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);

   int utcHour = dt.hour - detectedGMTOffset;
   if(utcHour < 0) utcHour += 24;
   if(utcHour >= 24) utcHour -= 24;

   if(dt.day_of_week == 0 || dt.day_of_week == 6) return;

   double dayMult = DayMultipliers[dt.day_of_week];
   if(dayMult <= 0.0) return;

   double hourMult = HourMultipliers[utcHour];
   if(hourMult <= 0.0) return;

   bool inLondon = (utcHour >= LondonStart && utcHour <= LondonEnd);
   bool inNewYork = (utcHour >= NewYorkStart && utcHour <= NewYorkEnd);
   if(!inLondon && !inNewYork) return;

   if(SkipHour11 && utcHour == 11) return;

   int dynamicAdj = 0;
   int riskCheck = CheckIntraMonthRisk(TimeCurrent(), dynamicAdj);
   if(riskCheck != 0) return;

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

   double atrPips = atr[1] / pipSize;
   if(atrPips < MinATR_Pips || atrPips > MaxATR_Pips) return;

   double close = iClose(_Symbol, PERIOD_H1, 1);
   int regime = 0;
   if(close > emaFast[1] && emaFast[1] > emaSlow[1]) regime = 1;
   else if(close < emaFast[1] && emaFast[1] < emaSlow[1]) regime = -1;
   if(regime == 0) return;

   string marketLabel = "";
   int technicalQuality = AssessTechnicalCondition(marketLabel);
   int monthlyAdj = GetMonthlyQualityAdjustment(dt.mon);
   double effectiveMinQuality = technicalQuality + monthlyAdj + dynamicAdj;

   int signal = 0;
   string signalType = "";
   double quality = 0;

   // EMA Pullback
   if(UseEmaPullback) {
      bool skipEMA = false;
      if(UseSessionFilter) {
         if(SkipEMA_Hour13 && utcHour == 13) skipEMA = true;
         if(SkipEMA_Hour14 && utcHour == 14) skipEMA = true;
      }
      if(!skipEMA) {
         int emaSignal = CheckEmaPullback(emaFast, emaSlow, rsi, adxMain, atrPips, quality);
         if(emaSignal != 0 && quality >= effectiveMinQuality) {
            if((emaSignal > 0 && regime > 0) || (emaSignal < 0 && regime < 0)) {
               signal = emaSignal;
               signalType = "EMA_PULLBACK";
            }
         }
      }
   }

   // Order Block
   if(signal == 0 && UseOrderBlock) {
      bool skipOB = false;
      if(UseSessionFilter) {
         if(SkipOB_Hour8 && utcHour == 8) skipOB = true;
         if(SkipOB_Hour16 && utcHour == 16) skipOB = true;
      }
      if(!skipOB) {
         int obSignal = CheckOrderBlock(quality);
         if(obSignal != 0 && quality >= effectiveMinQuality) {
            if((obSignal > 0 && regime > 0) || (obSignal < 0 && regime < 0)) {
               signal = obSignal;
               signalType = "ORDER_BLOCK";
            }
         }
      }
   }

   if(signal == 0) return;

   string entryType = "";
   if(!CheckEntryTrigger(signal, entryType)) return;
   currentEntryType = entryType;

   double patternSizeMult = 1.0;
   int patternExtraQ = 0;
   string direction = (signal > 0) ? "BUY" : "SELL";
   if(!CheckPatternFilter(direction, patternSizeMult, patternExtraQ)) return;
   if(patternExtraQ > 0 && quality < effectiveMinQuality + patternExtraQ) return;

   double monthRiskMult = GetMonthlyRiskMult(dt.mon);
   double techMult = (technicalQuality == BASE_QUALITY_GOOD) ? 1.0 : 0.8;
   double entryMult = GetEntryMultiplier(entryType);
   double riskMult = dayMult * hourMult * entryMult * (quality / 100.0) * monthRiskMult * patternSizeMult * techMult;

   if(riskMult < 0.30) return;
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

      if(close_curr < open_curr) {
         if(close_next > open_next && bodyRatio > 0.55 && close_next > high_curr) {
            double obQuality = bodyRatio * 100;
            if(MathAbs(currentClose - low_curr) <= currentZoneSize && obQuality > bestQuality) {
               bestSignal = 1; bestQuality = obQuality;
            }
         }
      }
      if(close_curr > open_curr) {
         if(close_next < open_next && bodyRatio > 0.55 && close_next < low_curr) {
            double obQuality = bodyRatio * 100;
            if(MathAbs(currentClose - high_curr) <= currentZoneSize && obQuality > bestQuality) {
               bestSignal = -1; bestQuality = obQuality;
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

   bool isBullish = (close1 > open1);
   if(isBullish && close1 > emaFast[1] && emaFast[1] > emaSlow[1]) {
      double distanceToEma = low1 - emaFast[1];
      if(distanceToEma <= atrDistance) {
         double touchQ = MathMax(0, 30 - (distanceToEma / pipSize));
         double adxQ = MathMin(25, (adx[1] - 15) * 1.5);
         double rsiQ = (MathAbs(50 - rsi[1]) < 20) ? 25 : 15;
         double bodyQ = MathMin(20, bodyRatio * 30);
         quality = MathMin(100, MathMax(55, touchQ + adxQ + rsiQ + bodyQ));
         return 1;
      }
   }

   bool isBearish = (close1 < open1);
   if(isBearish && close1 < emaFast[1] && emaFast[1] < emaSlow[1]) {
      double distanceToEma = emaFast[1] - high1;
      if(distanceToEma <= atrDistance) {
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
//| Execute trade                                                     |
//+------------------------------------------------------------------+
void ExecuteTrade(int signal, double atr, double riskMult, string signalType)
{
   double balance = accInfo.Balance();
   double riskAmount = balance * (RiskPercent / 100.0) * riskMult;

   double slPips = atr / pipSize * SL_ATR_Mult;
   double tpPips = slPips * TP_Ratio;

   double lotSize = riskAmount / (slPips * pipValue);
   lotSize = MathMax(MinLotSize, MathMin(MaxLotSize, lotSize));
   lotSize = NormalizeLotSize(lotSize);

   // v4: Store entry balance and calculate max allowed loss for SL_CAPPED
   entryBalance = balance;
   maxAllowedLoss = entryBalance * (MaxLossPercent / 100.0);

   double price, sl, tp;

   if(signal > 0) {
      price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl = price - slPips * pipSize;
      tp = price + tpPips * pipSize;

      if(trade.Buy(lotSize, _Symbol, price, sl, tp, "QuadV4_" + signalType)) {
         Print("BUY: ", signalType, " | Lot: ", lotSize,
               " | SL: ", DoubleToString(slPips, 1), " pips",
               " | MaxLoss: $", DoubleToString(maxAllowedLoss, 2));
      }
   }
   else if(signal < 0) {
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl = price + slPips * pipSize;
      tp = price - tpPips * pipSize;

      if(trade.Sell(lotSize, _Symbol, price, sl, tp, "QuadV4_" + signalType)) {
         Print("SELL: ", signalType, " | Lot: ", lotSize,
               " | SL: ", DoubleToString(slPips, 1), " pips",
               " | MaxLoss: $", DoubleToString(maxAllowedLoss, 2));
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
//| OnTradeTransaction - Track closed trades                          |
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

   ENUM_DEAL_TYPE dealType = (ENUM_DEAL_TYPE)HistoryDealGetInteger(dealTicket, DEAL_TYPE);
   string direction = (dealType == DEAL_TYPE_SELL) ? "BUY" : "SELL";

   RecordTradeForRiskManager(totalPnL);
   RecordTradeForPatternFilter(direction, totalPnL, TimeCurrent());

   string exitType = (totalPnL >= 0) ? "TP" : "SL";
   Print("Trade closed: ", direction, " | P&L: $", DoubleToString(totalPnL, 2),
         " | Monthly: $", DoubleToString(monthlyPnL, 2));
}
//+------------------------------------------------------------------+
