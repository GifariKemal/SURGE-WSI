//+------------------------------------------------------------------+
//|                                    GBPUSD_H1_QuadLayer_v69.mq5   |
//|                                    SURGE-WSI Trading System      |
//|                                    Version 6.9 - Quad Layer      |
//+------------------------------------------------------------------+
#property copyright "SURGE-WSI"
#property link      "https://github.com/surge-wsi"
#property version   "6.93"
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

input group "=== Session Filter ==="
input bool     UseSessionFilter = true;     // Enable Session Filter
input bool     SkipHour11 = true;           // Skip Hour 11 (low WR)
input bool     SkipOB_Hour8 = true;         // Skip OrderBlock at Hour 8
input bool     SkipOB_Hour16 = true;        // Skip OrderBlock at Hour 16
input bool     SkipEMA_Hour13 = true;       // Skip EMA Pullback Hour 13
input bool     SkipEMA_Hour14 = true;       // Skip EMA Pullback Hour 14

input group "=== Trading Hours (UTC) ==="
input bool     AutoDetectGMT = true;        // Auto-detect Broker GMT Offset
input int      GMTOffset = 0;               // Manual GMT Offset (if AutoDetect=false)
input int      LondonStart = 8;             // London Session Start (UTC)
input int      LondonEnd = 10;              // London Session End (UTC)
input int      NewYorkStart = 13;           // New York Session Start (UTC)
input int      NewYorkEnd = 17;             // New York Session End

input group "=== Magic Number ==="
input int      MagicNumber = 69001;         // EA Magic Number

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

//+------------------------------------------------------------------+
//| Check if running in Strategy Tester                               |
//+------------------------------------------------------------------+
bool IsBacktest()
{
   return MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_OPTIMIZATION);
}

//+------------------------------------------------------------------+
//| Detect broker GMT offset                                          |
//+------------------------------------------------------------------+
int DetectGMTOffset()
{
   // IMPORTANT: TimeGMT() may not work correctly in Strategy Tester
   // In backtests, it often returns the same as TimeCurrent()
   // So we need to handle this case

   if(IsBacktest())
   {
      // In Strategy Tester, TimeGMT() is unreliable
      // Most brokers use GMT+2 (or GMT+3 during DST)
      // Default to GMT+2 which is common for European brokers
      Print("WARNING: Running in Strategy Tester - GMT auto-detect may be inaccurate");
      Print("RECOMMENDATION: Set AutoDetectGMT=false and manually configure GMTOffset");
      Print("Common broker offsets: ICMarkets/Pepperstone=GMT+2/+3, XM=GMT+2/+3");

      // Try to detect anyway, but use fallback if it returns 0
      datetime gmtTime = TimeGMT();
      datetime serverTime = TimeCurrent();
      int diffSeconds = (int)(serverTime - gmtTime);
      int diffHours = diffSeconds / 3600;

      // If detection returns 0, assume GMT+2 (most common)
      if(diffHours == 0)
      {
         Print("Auto-detect returned 0, using fallback GMT+2 (common for most brokers)");
         return 2;
      }
      return diffHours;
   }

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

   // Detect or use manual GMT offset
   if(AutoDetectGMT)
   {
      detectedGMTOffset = DetectGMTOffset();
      Print("GMT Offset AUTO-DETECTED: GMT+", detectedGMTOffset);
   }
   else
   {
      detectedGMTOffset = GMTOffset;
      Print("GMT Offset MANUAL: GMT+", detectedGMTOffset);
   }

   // Log important time info
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int utcHour = dt.hour - detectedGMTOffset;
   if(utcHour < 0) utcHour += 24;
   if(utcHour >= 24) utcHour -= 24;

   Print("Server Time: ", TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES),
         " | UTC Hour: ", utcHour,
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

   Print("=== GBPUSD H1 QuadLayer v6.9 initialized ===");
   Print("Risk: ", RiskPercent, "% | Max Loss (SL_CAPPED): ", MaxLossPercent, "% | SL: ", SL_ATR_Mult, "x ATR | TP: ", TP_Ratio, ":1");
   Print("Trading Hours (UTC): London ", LondonStart, "-", LondonEnd, " | NY ", NewYorkStart, "-", NewYorkEnd);
   Print("Day Multipliers: Mon=1.0, Tue=0.9, Wed=1.0, Thu=0.8, Fri=0.3");

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

   Print("GBPUSD H1 QuadLayer v6.9 deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // Only trade on new bar
   static datetime lastBar = 0;
   datetime currentBar = iTime(_Symbol, PERIOD_H1, 0);
   if(lastBar == currentBar) return;
   lastBar = currentBar;

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

   // Check for entry signals
   int signal = 0;
   string signalType = "";
   double quality = 0;

   // Entry Signal 1: Order Block
   if(UseOrderBlock)
   {
      int obSignal = CheckOrderBlock(quality);
      if(obSignal != 0 && quality >= MinQuality)
      {
         // Session filter for Order Block (using UTC hour)
         if(UseSessionFilter)
         {
            if(SkipOB_Hour8 && utcHour == 8)
            {
               if(DebugMode) Print("DEBUG: ORDER_BLOCK filtered by Session POI (Hour 8)");
               obSignal = 0;
            }
            if(SkipOB_Hour16 && utcHour == 16)
            {
               if(DebugMode) Print("DEBUG: ORDER_BLOCK filtered by Session POI (Hour 16)");
               obSignal = 0;
            }
         }

         if(obSignal != 0 && ((obSignal > 0 && regime > 0) || (obSignal < 0 && regime < 0)))
         {
            signal = obSignal;
            signalType = "ORDER_BLOCK";
         }
         else if(obSignal != 0 && DebugMode)
         {
            Print("DEBUG: ORDER_BLOCK rejected - regime mismatch (signal=", obSignal, ", regime=", regime, ")");
         }
      }
   }

   // Entry Signal 2: EMA Pullback
   if(signal == 0 && UseEmaPullback)
   {
      int emaSignal = CheckEmaPullback(emaFast, emaSlow, rsi, adxMain, atrPips, quality);
      if(emaSignal != 0 && quality >= MinQuality)
      {
         // Session filter for EMA Pullback (using UTC hour)
         if(UseSessionFilter)
         {
            if(SkipEMA_Hour13 && utcHour == 13)
            {
               if(DebugMode) Print("DEBUG: EMA_PULLBACK filtered by Session POI (Hour 13)");
               emaSignal = 0;
            }
            if(SkipEMA_Hour14 && utcHour == 14)
            {
               if(DebugMode) Print("DEBUG: EMA_PULLBACK filtered by Session POI (Hour 14)");
               emaSignal = 0;
            }
         }

         if(emaSignal != 0 && ((emaSignal > 0 && regime > 0) || (emaSignal < 0 && regime < 0)))
         {
            signal = emaSignal;
            signalType = "EMA_PULLBACK";
         }
         else if(emaSignal != 0 && DebugMode)
         {
            Print("DEBUG: EMA_PULLBACK rejected - regime mismatch (signal=", emaSignal, ", regime=", regime, ")");
         }
      }
   }

   // Execute trade if signal found
   if(signal != 0)
   {
      double riskMult = dayMult * hourMult * (quality / 100.0);
      ExecuteTrade(signal, atr[1], riskMult, signalType);
   }
}

//+------------------------------------------------------------------+
//| Check for Order Block signal                                      |
//+------------------------------------------------------------------+
int CheckOrderBlock(double &quality)
{
   double open1 = iOpen(_Symbol, PERIOD_H1, 1);
   double close1 = iClose(_Symbol, PERIOD_H1, 1);
   double high1 = iHigh(_Symbol, PERIOD_H1, 1);
   double low1 = iLow(_Symbol, PERIOD_H1, 1);

   double open2 = iOpen(_Symbol, PERIOD_H1, 2);
   double close2 = iClose(_Symbol, PERIOD_H1, 2);
   double high2 = iHigh(_Symbol, PERIOD_H1, 2);
   double low2 = iLow(_Symbol, PERIOD_H1, 2);

   double range1 = high1 - low1;
   if(range1 < 0.0003) return 0;

   double body1 = MathAbs(close1 - open1);
   double bodyRatio = body1 / range1;

   // Bullish Order Block: Previous bearish, current bullish engulf
   if(close2 < open2) // Previous bearish
   {
      if(close1 > open1 && bodyRatio > 0.55 && close1 > high2)
      {
         quality = bodyRatio * 100;
         return 1; // BUY
      }
   }

   // Bearish Order Block: Previous bullish, current bearish engulf
   if(close2 > open2) // Previous bullish
   {
      if(close1 < open1 && bodyRatio > 0.55 && close1 < low2)
      {
         quality = bodyRatio * 100;
         return -1; // SELL
      }
   }

   return 0;
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

      if(trade.Buy(lotSize, _Symbol, price, sl, tp, "QuadLayer_" + signalType))
      {
         Print("BUY executed: ", signalType, " | Lot: ", lotSize, " | SL: ", slPips, " pips | TP: ", tpPips, " pips");
      }
   }
   else if(signal < 0) // SELL
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl = price + slPips * pipSize;
      tp = price - tpPips * pipSize;

      if(trade.Sell(lotSize, _Symbol, price, sl, tp, "QuadLayer_" + signalType))
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
