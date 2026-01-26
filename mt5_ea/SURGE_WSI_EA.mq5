//+------------------------------------------------------------------+
//|                                            SURGE_WSI_EA.mq5      |
//|                                    Copyright 2025, SURIOTA Team  |
//|              SURGE-WSI Simplified EA - Pure Price Action         |
//|                  Supply/Demand + Kill Zone + Partial TP          |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, SURIOTA Team"
#property link      "https://github.com/suriota"
#property version   "1.00"
#property description "SURGE-WSI Simplified Trading EA"
#property description "Based on Python system but without ML/HMM"
#property description "Features: S/D Zones, Kill Zone, Partial TP, December Filter"
#property description "Optimized for GBPUSD H1 timeframe"

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Enumerations                                                      |
//+------------------------------------------------------------------+
enum ENUM_SD_ZONE_TYPE {
   ZONE_RBD,    // Rally-Base-Drop = Supply
   ZONE_DBR,    // Drop-Base-Rally = Demand
   ZONE_RBR,    // Rally-Base-Rally = Continuation Demand
   ZONE_DBD     // Drop-Base-Drop = Continuation Supply
};

enum ENUM_ZONE_STATUS {
   ZONE_FRESH,      // Never tested
   ZONE_TESTED,     // Tested 1 time
   ZONE_WEAK,       // Tested 2+ times
   ZONE_MITIGATED   // Broken through
};

enum ENUM_TRADE_MODE {
   MODE_AUTO,       // Full auto trading
   MODE_RECOVERY,   // Reduced lot size after losses
   MODE_SIGNAL_ONLY // Only log signals, no execution
};

//+------------------------------------------------------------------+
//| Input Parameters                                                  |
//+------------------------------------------------------------------+
input group "=== Risk Management ==="
input double   RiskPercent = 1.0;           // Risk per trade (%)
input double   RecoveryRiskPercent = 0.5;   // Risk in recovery mode (%)
input double   MinRRRatio = 1.5;            // Minimum Risk:Reward ratio
input int      MaxTradesPerDay = 3;         // Max trades per day
input double   MaxDrawdownPercent = 10.0;   // Max drawdown % to pause
input int      MaxConsecutiveLosses = 3;    // Enter recovery after N losses
input double   DailyLossLimit = 3.0;        // Daily loss limit %

input group "=== Zone Detection ==="
input int      SwingLookback = 5;           // Swing detection bars
input int      MaxBaseCandles = 5;          // Max base candles
input double   MinImbalanceRatio = 1.5;     // Min departure/base ratio
input int      MinQualityScore = 50;        // Min zone quality (0-100)
input int      MaxZoneAge = 200;            // Max zone age in bars
input int      MaxTestCount = 2;            // Max zone retest count

input group "=== Kill Zone Hours (UTC) ==="
input bool     UseKillZone = true;          // Only trade in Kill Zones
input int      LondonStart = 8;             // London session start
input int      LondonEnd = 12;              // London session end
input int      NYStart = 13;                // New York session start
input int      NYEnd = 17;                  // New York session end
input int      LondonCloseStart = 15;       // London close start
input int      LondonCloseEnd = 17;         // London close end

input group "=== December Filter ==="
input bool     UseDecemberFilter = true;    // Enable December restrictions
input int      DecemberSignalOnlyDay = 1;   // Signal-only mode starts
input int      DecemberMonitoringDay = 15;  // Full pause starts

input group "=== Entry Confirmation ==="
input double   RejectionWickRatio = 0.5;    // Rejection wick > 50% of range
input double   PinBarWickRatio = 2.0;       // Pin bar wick/body ratio

input group "=== Partial TP Settings ==="
input bool     UsePartialTP = true;         // Enable partial TP
input double   TP1_RR = 1.0;                // TP1 at R:R (close 50%)
input double   TP1_Percent = 50.0;          // TP1 close percent
input double   TP2_RR = 2.0;                // TP2 at R:R (close 30%)
input double   TP2_Percent = 30.0;          // TP2 close percent
input double   TP3_RR = 3.0;                // TP3 at R:R (close 20%)
input bool     UseBreakeven = true;         // Move SL to BE after TP1
input bool     UseTrailingAfterTP2 = true;  // Trail after TP2

input group "=== Trailing Stop ==="
input double   TrailingATRMultiplier = 1.5; // ATR multiplier for trailing
input int      ATRPeriod = 14;              // ATR period

input group "=== Display ==="
input bool     ShowZones = true;
input bool     ShowSignals = true;
input color    DemandZoneColor = clrDodgerBlue;
input color    SupplyZoneColor = clrCrimson;
input int      MagicNumber = 2025125;       // EA Magic Number

//+------------------------------------------------------------------+
//| Data Structures                                                   |
//+------------------------------------------------------------------+
struct SupplyDemandZone {
   ENUM_SD_ZONE_TYPE type;
   ENUM_ZONE_STATUS status;
   double   zoneHigh;
   double   zoneLow;
   double   proximalLine;
   double   distalLine;
   int      qualityScore;
   int      testCount;
   bool     isBullish;
   datetime formationTime;
   int      formationIdx;
   int      age;
};

struct TradeStats {
   int      consecutiveLosses;
   int      consecutiveWins;
   double   dailyPL;
   int      dailyTradeCount;
   datetime lastTradeDate;
};

struct PositionState {
   double   entryPrice;
   double   originalSL;
   double   originalTP;
   datetime entryTime;
   bool     tp1Taken;
   bool     tp2Taken;
   bool     breakevenSet;
   double   initialVolume;
};

//+------------------------------------------------------------------+
//| Global Variables                                                  |
//+------------------------------------------------------------------+
CTrade trade;

SupplyDemandZone zones[];
TradeStats stats;
PositionState posState;

ENUM_TRADE_MODE currentMode = MODE_AUTO;
int todayTrades = 0;
datetime lastBarTime = 0;

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_IOC);

   // Initialize stats
   stats.consecutiveLosses = 0;
   stats.consecutiveWins = 0;
   stats.dailyPL = 0;
   stats.dailyTradeCount = 0;
   stats.lastTradeDate = 0;

   // Initialize position state
   ResetPositionState();

   Print("==========================================================");
   Print("SURGE-WSI EA v1.00 - Simplified Pure Price Action");
   Print("==========================================================");
   Print("Strategy: Supply/Demand Zones + Kill Zone Filter");
   Print("Partial TP: ", TP1_Percent, "% @ ", TP1_RR, "R / ",
         TP2_Percent, "% @ ", TP2_RR, "R / 20% @ ", TP3_RR, "R");
   Print("December Filter: ", UseDecemberFilter ? "ENABLED" : "DISABLED");
   Print("==========================================================");

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   ObjectsDeleteAll(0, "SWSI_");
   Print("SURGE-WSI EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // Manage open position first
   if(PositionSelect(_Symbol)) {
      ManageOpenPosition();
   }

   // Only process on new bar
   datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(currentBarTime == lastBarTime)
      return;
   lastBarTime = currentBarTime;

   // Daily reset
   ResetDailyCounter();

   // Update trade mode
   UpdateTradeMode();

   // December filter check
   if(IsDecemberMonitoring()) {
      return; // Full pause - no signals, no trading
   }

   // Update zones
   UpdateZonesAge();
   DetectSupplyDemandZones();
   UpdateZoneStatus();

   // Draw zones if enabled
   if(ShowZones) DrawZones();

   // Check if we can trade
   if(PositionSelect(_Symbol)) return;  // Already have position
   if(todayTrades >= MaxTradesPerDay) return;
   if(CheckDrawdownLimit()) return;
   if(stats.dailyPL <= -DailyLossLimit) return;

   // Kill zone check
   if(UseKillZone && !IsInKillZone()) return;

   // December signal-only mode
   bool signalOnly = IsDecemberSignalOnly() || (currentMode == MODE_SIGNAL_ONLY);

   // Look for entry
   CheckForEntry(signalOnly);
}

//+------------------------------------------------------------------+
//| Reset Position State                                              |
//+------------------------------------------------------------------+
void ResetPositionState()
{
   posState.entryPrice = 0;
   posState.originalSL = 0;
   posState.originalTP = 0;
   posState.entryTime = 0;
   posState.tp1Taken = false;
   posState.tp2Taken = false;
   posState.breakevenSet = false;
   posState.initialVolume = 0;
}

//+------------------------------------------------------------------+
//| Check December Monitoring (Full Pause)                            |
//+------------------------------------------------------------------+
bool IsDecemberMonitoring()
{
   if(!UseDecemberFilter) return false;

   MqlDateTime dt;
   TimeCurrent(dt);

   if(dt.mon == 12 && dt.day >= DecemberMonitoringDay) {
      return true;  // Dec 15-31: Full pause
   }
   return false;
}

//+------------------------------------------------------------------+
//| Check December Signal-Only Mode                                   |
//+------------------------------------------------------------------+
bool IsDecemberSignalOnly()
{
   if(!UseDecemberFilter) return false;

   MqlDateTime dt;
   TimeCurrent(dt);

   if(dt.mon == 12 && dt.day >= DecemberSignalOnlyDay && dt.day < DecemberMonitoringDay) {
      return true;  // Dec 1-14: Signal only
   }
   return false;
}

//+------------------------------------------------------------------+
//| Update Trade Mode                                                 |
//+------------------------------------------------------------------+
void UpdateTradeMode()
{
   ENUM_TRADE_MODE previousMode = currentMode;

   // Check for recovery mode trigger
   if(stats.consecutiveLosses >= MaxConsecutiveLosses) {
      currentMode = MODE_RECOVERY;
   } else if(stats.consecutiveWins >= 2 && currentMode == MODE_RECOVERY) {
      // Exit recovery after 2 consecutive wins
      currentMode = MODE_AUTO;
      Print("SURGE-WSI: Exiting RECOVERY mode after ", stats.consecutiveWins, " wins");
   }

   // Log mode changes
   if(previousMode != currentMode) {
      string modeName = (currentMode == MODE_AUTO ? "AUTO" :
                        (currentMode == MODE_RECOVERY ? "RECOVERY" : "SIGNAL_ONLY"));
      Print("SURGE-WSI: Trade mode changed to ", modeName);
   }
}

//+------------------------------------------------------------------+
//| Check if in Kill Zone                                             |
//+------------------------------------------------------------------+
bool IsInKillZone()
{
   MqlDateTime dt;
   TimeToStruct(TimeGMT(), dt);
   int hour = dt.hour;

   // London session
   if(hour >= LondonStart && hour < LondonEnd) return true;

   // New York session
   if(hour >= NYStart && hour < NYEnd) return true;

   // London close (overlap)
   if(hour >= LondonCloseStart && hour < LondonCloseEnd) return true;

   return false;
}

//+------------------------------------------------------------------+
//| Get Session Quality Score                                         |
//+------------------------------------------------------------------+
int GetSessionQuality()
{
   MqlDateTime dt;
   TimeToStruct(TimeGMT(), dt);
   int hour = dt.hour;

   // Best hours based on volatility analysis
   if(hour >= 8 && hour <= 11) return 100;   // London morning
   if(hour >= 13 && hour <= 15) return 90;   // NY open overlap
   if(hour >= 15 && hour < 17) return 70;    // London close
   if(hour >= 7 && hour < 8) return 50;      // Pre-London
   if(hour >= 17 && hour < 19) return 40;    // Post-overlap

   return 20;  // Asian/off-hours
}

//+------------------------------------------------------------------+
//| Get Average True Range                                            |
//+------------------------------------------------------------------+
double GetATR(int period = 0, int shift = 1)
{
   if(period == 0) period = ATRPeriod;

   double totalRange = 0;
   for(int i = shift; i < shift + period; i++) {
      double high = iHigh(_Symbol, PERIOD_CURRENT, i);
      double low = iLow(_Symbol, PERIOD_CURRENT, i);
      double prevClose = iClose(_Symbol, PERIOD_CURRENT, i + 1);

      double tr1 = high - low;
      double tr2 = MathAbs(high - prevClose);
      double tr3 = MathAbs(low - prevClose);

      totalRange += MathMax(tr1, MathMax(tr2, tr3));
   }

   return totalRange / period;
}

//+------------------------------------------------------------------+
//| Check if candle is consolidation/base candle                      |
//+------------------------------------------------------------------+
bool IsBaseCandle(int idx, double atr)
{
   double open = iOpen(_Symbol, PERIOD_CURRENT, idx);
   double close = iClose(_Symbol, PERIOD_CURRENT, idx);
   double high = iHigh(_Symbol, PERIOD_CURRENT, idx);
   double low = iLow(_Symbol, PERIOD_CURRENT, idx);

   double body = MathAbs(close - open);
   double range = high - low;

   if(range <= 0) return false;

   double bodyRatio = body / range;
   return (bodyRatio < 0.5 && range < atr * 0.8);
}

//+------------------------------------------------------------------+
//| Check for Rejection Candle                                        |
//+------------------------------------------------------------------+
bool IsRejectionCandle(int idx, bool checkBullish)
{
   double open = iOpen(_Symbol, PERIOD_CURRENT, idx);
   double close = iClose(_Symbol, PERIOD_CURRENT, idx);
   double high = iHigh(_Symbol, PERIOD_CURRENT, idx);
   double low = iLow(_Symbol, PERIOD_CURRENT, idx);

   double body = MathAbs(close - open);
   double range = high - low;

   if(range <= 0) return false;

   double upperWick = high - MathMax(open, close);
   double lowerWick = MathMin(open, close) - low;

   if(checkBullish) {
      // Bullish rejection: long lower wick
      return (lowerWick > range * RejectionWickRatio);
   } else {
      // Bearish rejection: long upper wick
      return (upperWick > range * RejectionWickRatio);
   }
}

//+------------------------------------------------------------------+
//| Check for Pin Bar                                                 |
//+------------------------------------------------------------------+
bool IsPinBar(int idx, bool checkBullish)
{
   double open = iOpen(_Symbol, PERIOD_CURRENT, idx);
   double close = iClose(_Symbol, PERIOD_CURRENT, idx);
   double high = iHigh(_Symbol, PERIOD_CURRENT, idx);
   double low = iLow(_Symbol, PERIOD_CURRENT, idx);

   double body = MathAbs(close - open);
   double range = high - low;

   if(range <= 0 || body <= 0) return false;

   double upperWick = high - MathMax(open, close);
   double lowerWick = MathMin(open, close) - low;

   if(checkBullish) {
      // Bullish pin bar: long lower wick, small upper wick
      bool longLowerWick = lowerWick > body * PinBarWickRatio;
      bool shortUpperWick = upperWick < body;
      bool bodyInUpper = (MathMin(open, close) > low + range * 0.6);
      return longLowerWick && shortUpperWick && bodyInUpper;
   } else {
      // Bearish pin bar: long upper wick, small lower wick
      bool longUpperWick = upperWick > body * PinBarWickRatio;
      bool shortLowerWick = lowerWick < body;
      bool bodyInLower = (MathMax(open, close) < high - range * 0.6);
      return longUpperWick && shortLowerWick && bodyInLower;
   }
}

//+------------------------------------------------------------------+
//| Detect Supply/Demand Zones                                        |
//+------------------------------------------------------------------+
void DetectSupplyDemandZones()
{
   // Keep valid zones, remove mitigated/old ones
   SupplyDemandZone tempZones[];
   int validCount = 0;

   for(int i = 0; i < ArraySize(zones); i++) {
      if(zones[i].status != ZONE_MITIGATED && zones[i].age < MaxZoneAge) {
         ArrayResize(tempZones, validCount + 1);
         tempZones[validCount] = zones[i];
         validCount++;
      }
   }

   ArrayResize(zones, 0);
   for(int i = 0; i < validCount; i++) {
      int size = ArraySize(zones);
      ArrayResize(zones, size + 1);
      zones[size] = tempZones[i];
   }

   double atr = GetATR();

   // Scan for new zones
   for(int i = 5; i < 50; i++) {
      // Check for existing zone at this location
      bool exists = false;
      for(int j = 0; j < ArraySize(zones); j++) {
         if(MathAbs(zones[j].formationIdx - i) < 3) {
            exists = true;
            break;
         }
      }
      if(exists) continue;

      // Look for base candles
      if(!IsBaseCandle(i, atr)) continue;

      // Check for RBD (Supply) pattern
      SupplyDemandZone supplyZone;
      if(DetectRBDPattern(i, atr, supplyZone)) {
         if(supplyZone.qualityScore >= MinQualityScore) {
            int size = ArraySize(zones);
            ArrayResize(zones, size + 1);
            zones[size] = supplyZone;
         }
      }

      // Check for DBR (Demand) pattern
      SupplyDemandZone demandZone;
      if(DetectDBRPattern(i, atr, demandZone)) {
         if(demandZone.qualityScore >= MinQualityScore) {
            int size = ArraySize(zones);
            ArrayResize(zones, size + 1);
            zones[size] = demandZone;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Detect RBD (Supply) Pattern                                       |
//+------------------------------------------------------------------+
bool DetectRBDPattern(int baseIdx, double atr, SupplyDemandZone &zone)
{
   // Need rally before base
   double baseHigh = iHigh(_Symbol, PERIOD_CURRENT, baseIdx);
   double baseLow = iLow(_Symbol, PERIOD_CURRENT, baseIdx);

   // Find base range (can be 1-MaxBaseCandles)
   int baseStart = baseIdx;
   int baseEnd = baseIdx;
   double zoneHigh = baseHigh;
   double zoneLow = baseLow;

   for(int i = baseIdx; i < baseIdx + MaxBaseCandles && i < 100; i++) {
      if(IsBaseCandle(i, atr)) {
         baseEnd = i;
         zoneHigh = MathMax(zoneHigh, iHigh(_Symbol, PERIOD_CURRENT, i));
         zoneLow = MathMin(zoneLow, iLow(_Symbol, PERIOD_CURRENT, i));
      } else {
         break;
      }
   }

   // Check for rally before base (price coming up)
   double rallyLow = DBL_MAX;
   for(int i = baseEnd + 1; i < baseEnd + 10 && i < 100; i++) {
      rallyLow = MathMin(rallyLow, iLow(_Symbol, PERIOD_CURRENT, i));
   }

   if(rallyLow >= zoneLow) return false;  // No rally

   // Check for drop after base
   double dropLow = DBL_MAX;
   for(int i = baseStart - 1; i >= MathMax(0, baseStart - 10); i--) {
      dropLow = MathMin(dropLow, iLow(_Symbol, PERIOD_CURRENT, i));
   }

   if(dropLow >= zoneLow) return false;  // No drop

   // Check imbalance ratio
   double baseHeight = zoneHigh - zoneLow;
   double dropDistance = zoneLow - dropLow;

   if(baseHeight <= 0 || dropDistance / baseHeight < MinImbalanceRatio) return false;

   // Calculate quality score (0-100)
   int quality = 50;

   // Departure strength
   if(dropDistance > atr * 3) quality += 20;
   else if(dropDistance > atr * 2) quality += 10;

   // Base tightness
   int baseCandles = baseEnd - baseStart + 1;
   if(baseCandles == 1) quality += 15;
   else if(baseCandles == 2) quality += 10;
   else if(baseCandles == 3) quality += 5;

   // Session quality
   quality += GetSessionQuality() / 10;

   zone.type = ZONE_RBD;
   zone.status = ZONE_FRESH;
   zone.zoneHigh = zoneHigh;
   zone.zoneLow = zoneLow;
   zone.proximalLine = zoneLow;
   zone.distalLine = zoneHigh;
   zone.isBullish = false;
   zone.testCount = 0;
   zone.formationTime = iTime(_Symbol, PERIOD_CURRENT, baseStart);
   zone.formationIdx = baseStart;
   zone.age = 0;
   zone.qualityScore = MathMin(100, quality);

   return true;
}

//+------------------------------------------------------------------+
//| Detect DBR (Demand) Pattern                                       |
//+------------------------------------------------------------------+
bool DetectDBRPattern(int baseIdx, double atr, SupplyDemandZone &zone)
{
   double baseHigh = iHigh(_Symbol, PERIOD_CURRENT, baseIdx);
   double baseLow = iLow(_Symbol, PERIOD_CURRENT, baseIdx);

   // Find base range
   int baseStart = baseIdx;
   int baseEnd = baseIdx;
   double zoneHigh = baseHigh;
   double zoneLow = baseLow;

   for(int i = baseIdx; i < baseIdx + MaxBaseCandles && i < 100; i++) {
      if(IsBaseCandle(i, atr)) {
         baseEnd = i;
         zoneHigh = MathMax(zoneHigh, iHigh(_Symbol, PERIOD_CURRENT, i));
         zoneLow = MathMin(zoneLow, iLow(_Symbol, PERIOD_CURRENT, i));
      } else {
         break;
      }
   }

   // Check for drop before base
   double dropHigh = 0;
   for(int i = baseEnd + 1; i < baseEnd + 10 && i < 100; i++) {
      dropHigh = MathMax(dropHigh, iHigh(_Symbol, PERIOD_CURRENT, i));
   }

   if(dropHigh <= zoneHigh) return false;  // No drop

   // Check for rally after base
   double rallyHigh = 0;
   for(int i = baseStart - 1; i >= MathMax(0, baseStart - 10); i--) {
      rallyHigh = MathMax(rallyHigh, iHigh(_Symbol, PERIOD_CURRENT, i));
   }

   if(rallyHigh <= zoneHigh) return false;  // No rally

   // Check imbalance ratio
   double baseHeight = zoneHigh - zoneLow;
   double rallyDistance = rallyHigh - zoneHigh;

   if(baseHeight <= 0 || rallyDistance / baseHeight < MinImbalanceRatio) return false;

   // Calculate quality score
   int quality = 50;

   if(rallyDistance > atr * 3) quality += 20;
   else if(rallyDistance > atr * 2) quality += 10;

   int baseCandles = baseEnd - baseStart + 1;
   if(baseCandles == 1) quality += 15;
   else if(baseCandles == 2) quality += 10;
   else if(baseCandles == 3) quality += 5;

   quality += GetSessionQuality() / 10;

   zone.type = ZONE_DBR;
   zone.status = ZONE_FRESH;
   zone.zoneHigh = zoneHigh;
   zone.zoneLow = zoneLow;
   zone.proximalLine = zoneHigh;
   zone.distalLine = zoneLow;
   zone.isBullish = true;
   zone.testCount = 0;
   zone.formationTime = iTime(_Symbol, PERIOD_CURRENT, baseStart);
   zone.formationIdx = baseStart;
   zone.age = 0;
   zone.qualityScore = MathMin(100, quality);

   return true;
}

//+------------------------------------------------------------------+
//| Update Zone Status                                                |
//+------------------------------------------------------------------+
void UpdateZoneStatus()
{
   double currentPrice = iClose(_Symbol, PERIOD_CURRENT, 0);
   double atr = GetATR();

   for(int i = 0; i < ArraySize(zones); i++) {
      if(zones[i].status == ZONE_MITIGATED) continue;

      // Check if zone is mitigated (price broke through)
      if(zones[i].isBullish) {
         if(currentPrice < zones[i].distalLine - atr * 0.2) {
            zones[i].status = ZONE_MITIGATED;
         }
      } else {
         if(currentPrice > zones[i].distalLine + atr * 0.2) {
            zones[i].status = ZONE_MITIGATED;
         }
      }

      // Check if price is testing zone
      bool priceInZone = (currentPrice >= zones[i].zoneLow &&
                          currentPrice <= zones[i].zoneHigh);

      if(priceInZone && zones[i].status != ZONE_MITIGATED) {
         static datetime lastTestTime[];
         ArrayResize(lastTestTime, MathMax(ArraySize(lastTestTime), i + 1));

         datetime currentTime = TimeCurrent();
         if(currentTime - lastTestTime[i] > PeriodSeconds(PERIOD_CURRENT) * 5) {
            zones[i].testCount++;
            lastTestTime[i] = currentTime;

            if(zones[i].testCount == 1) zones[i].status = ZONE_TESTED;
            else if(zones[i].testCount >= 2) zones[i].status = ZONE_WEAK;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Update Zones Age                                                  |
//+------------------------------------------------------------------+
void UpdateZonesAge()
{
   for(int i = 0; i < ArraySize(zones); i++) {
      zones[i].age++;
   }
}

//+------------------------------------------------------------------+
//| Check for Entry                                                   |
//+------------------------------------------------------------------+
void CheckForEntry(bool signalOnly)
{
   double currentPrice = iClose(_Symbol, PERIOD_CURRENT, 0);
   double atr = GetATR();

   for(int i = 0; i < ArraySize(zones); i++) {
      if(zones[i].status == ZONE_MITIGATED) continue;
      if(zones[i].status == ZONE_WEAK) continue;
      if(zones[i].testCount > MaxTestCount) continue;
      if(zones[i].qualityScore < MinQualityScore) continue;
      if(zones[i].age > MaxZoneAge) continue;

      // Check if price is in zone
      bool priceInZone = (currentPrice >= zones[i].zoneLow - atr * 0.1 &&
                          currentPrice <= zones[i].zoneHigh + atr * 0.1);

      if(!priceInZone) continue;

      // Check for entry confirmation
      bool hasRejection = IsRejectionCandle(1, zones[i].isBullish);
      bool hasPinBar = IsPinBar(1, zones[i].isBullish);

      if(!hasRejection && !hasPinBar) continue;

      // Calculate SL and TP
      double zoneHeight = zones[i].zoneHigh - zones[i].zoneLow;
      double buffer = MathMax(zoneHeight * 0.3, atr * 0.5);

      double entry, sl, tp;

      if(zones[i].isBullish) {
         entry = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         sl = zones[i].distalLine - buffer;
         double risk = entry - sl;
         tp = entry + risk * MinRRRatio;
      } else {
         entry = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         sl = zones[i].distalLine + buffer;
         double risk = sl - entry;
         tp = entry - risk * MinRRRatio;
      }

      // Validate RR
      double risk = MathAbs(entry - sl);
      double reward = MathAbs(tp - entry);
      if(reward / risk < MinRRRatio) continue;

      // Signal only mode - just log
      if(signalOnly) {
         string signalType = zones[i].isBullish ? "BUY" : "SELL";
         string pattern = hasPinBar ? "Pin Bar" : "Rejection";
         Print("SURGE-WSI SIGNAL: ", signalType, " @ ", entry,
               " | SL: ", sl, " | TP: ", tp,
               " | Zone Q: ", zones[i].qualityScore,
               " | Pattern: ", pattern);
         if(ShowSignals) DrawSignal(zones[i].isBullish, entry, sl, tp);
         return;  // Signal logged, done
      }

      // Execute trade
      ExecuteTrade(zones[i], entry, sl, tp, hasPinBar);
      return;  // Only one trade per bar
   }
}

//+------------------------------------------------------------------+
//| Execute Trade                                                     |
//+------------------------------------------------------------------+
void ExecuteTrade(SupplyDemandZone &zone, double entry, double sl, double tp, bool isPinBar)
{
   // Determine risk based on mode
   double riskPercent = (currentMode == MODE_RECOVERY) ? RecoveryRiskPercent : RiskPercent;

   // Calculate lot size
   double lotSize = CalculateLotSize(MathAbs(entry - sl), riskPercent);
   if(lotSize <= 0) return;

   string pattern = isPinBar ? "PinBar" : "Reject";
   string modeName = (currentMode == MODE_AUTO ? "AUTO" : "RECOV");
   string comment = "SWSI_" + (zone.isBullish ? "Buy" : "Sell") + "_Q" +
                    IntegerToString(zone.qualityScore) + "_" + pattern + "_" + modeName;

   bool success = false;

   if(zone.isBullish) {
      success = trade.Buy(lotSize, _Symbol, entry, sl, tp, comment);
   } else {
      success = trade.Sell(lotSize, _Symbol, entry, sl, tp, comment);
   }

   if(success) {
      // Save position state for management
      posState.entryPrice = entry;
      posState.originalSL = sl;
      posState.originalTP = tp;
      posState.entryTime = TimeCurrent();
      posState.tp1Taken = false;
      posState.tp2Taken = false;
      posState.breakevenSet = false;
      posState.initialVolume = lotSize;

      todayTrades++;
      stats.dailyTradeCount++;

      Print("SURGE-WSI ", (zone.isBullish ? "BUY" : "SELL"),
            " | Entry: ", entry,
            " | SL: ", sl, " | TP: ", tp,
            " | Lot: ", lotSize,
            " | Zone Q: ", zone.qualityScore,
            " | Mode: ", modeName);

      if(ShowSignals) DrawSignal(zone.isBullish, entry, sl, tp);
   }
}

//+------------------------------------------------------------------+
//| Calculate Lot Size                                                |
//+------------------------------------------------------------------+
double CalculateLotSize(double slDistance, double riskPercent)
{
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = accountBalance * (riskPercent / 100.0);

   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

   if(tickSize == 0 || tickValue == 0) return 0;

   double slTicks = slDistance / tickSize;
   double lotSize = riskAmount / (slTicks * tickValue);

   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   lotSize = MathFloor(lotSize / lotStep) * lotStep;
   lotSize = MathMax(minLot, MathMin(maxLot, lotSize));

   return lotSize;
}

//+------------------------------------------------------------------+
//| Manage Open Position                                              |
//+------------------------------------------------------------------+
void ManageOpenPosition()
{
   if(!PositionSelect(_Symbol)) return;

   ulong positionMagic = PositionGetInteger(POSITION_MAGIC);
   if(positionMagic != MagicNumber) return;  // Not our position

   double positionProfit = PositionGetDouble(POSITION_PROFIT);
   double currentPrice = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ?
                         SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                         SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double positionSL = PositionGetDouble(POSITION_SL);
   double positionTP = PositionGetDouble(POSITION_TP);
   double positionEntry = PositionGetDouble(POSITION_PRICE_OPEN);
   double currentVolume = PositionGetDouble(POSITION_VOLUME);
   bool isBuy = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY;

   double risk = isBuy ? (positionEntry - posState.originalSL) : (posState.originalSL - positionEntry);
   if(risk <= 0) return;

   double currentRR = isBuy ? (currentPrice - positionEntry) / risk : (positionEntry - currentPrice) / risk;

   double atr = GetATR();
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   // TP1: Close 50% at 1:1 R:R
   if(UsePartialTP && !posState.tp1Taken && currentRR >= TP1_RR) {
      double closeVolume = posState.initialVolume * (TP1_Percent / 100.0);
      closeVolume = MathFloor(closeVolume / lotStep) * lotStep;
      closeVolume = MathMax(minLot, closeVolume);

      if(closeVolume < currentVolume) {
         if(trade.PositionClosePartial(_Symbol, closeVolume)) {
            posState.tp1Taken = true;
            Print("SURGE-WSI: TP1 hit (", TP1_Percent, "% @ ", TP1_RR, "R)");

            // Move to breakeven
            if(UseBreakeven && !posState.breakevenSet) {
               double newSL = positionEntry + (isBuy ? _Point * 5 : -_Point * 5);
               if((isBuy && newSL > positionSL) || (!isBuy && newSL < positionSL)) {
                  if(trade.PositionModify(_Symbol, newSL, positionTP)) {
                     posState.breakevenSet = true;
                     Print("SURGE-WSI: SL moved to breakeven");
                  }
               }
            }
         }
      }
   }

   // TP2: Close 30% at 2:1 R:R
   if(UsePartialTP && posState.tp1Taken && !posState.tp2Taken && currentRR >= TP2_RR) {
      // Recalculate based on remaining volume
      double remainingVolume = currentVolume;
      double closeVolume = posState.initialVolume * (TP2_Percent / 100.0);
      closeVolume = MathFloor(closeVolume / lotStep) * lotStep;
      closeVolume = MathMax(minLot, closeVolume);

      if(closeVolume < remainingVolume) {
         if(trade.PositionClosePartial(_Symbol, closeVolume)) {
            posState.tp2Taken = true;
            Print("SURGE-WSI: TP2 hit (", TP2_Percent, "% @ ", TP2_RR, "R)");
         }
      }
   }

   // Trailing stop after TP2
   if(UseTrailingAfterTP2 && posState.tp2Taken && currentRR >= TP2_RR) {
      double trailDistance = atr * TrailingATRMultiplier;
      double newSL;

      if(isBuy) {
         newSL = currentPrice - trailDistance;
         if(newSL > positionSL + _Point) {
            trade.PositionModify(_Symbol, newSL, positionTP);
         }
      } else {
         newSL = currentPrice + trailDistance;
         if(newSL < positionSL - _Point) {
            trade.PositionModify(_Symbol, newSL, positionTP);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check Drawdown Limit                                              |
//+------------------------------------------------------------------+
bool CheckDrawdownLimit()
{
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double accountEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   double drawdown = (accountBalance - accountEquity) / accountBalance * 100;

   if(drawdown >= MaxDrawdownPercent) {
      return true;
   }

   return false;
}

//+------------------------------------------------------------------+
//| Reset Daily Counter                                               |
//+------------------------------------------------------------------+
void ResetDailyCounter()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   datetime today = StringToTime(StringFormat("%04d.%02d.%02d", dt.year, dt.mon, dt.day));

   if(today != stats.lastTradeDate) {
      todayTrades = 0;
      stats.dailyPL = 0;
      stats.dailyTradeCount = 0;
      stats.lastTradeDate = today;

      // Weekly reset of consecutive losses
      if(dt.day_of_week == 1) {
         stats.consecutiveLosses = 0;
         Print("SURGE-WSI: Weekly reset - consecutive losses cleared");
      }
   }
}

//+------------------------------------------------------------------+
//| Handle Trade Events                                               |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
{
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD) {
      ulong dealTicket = trans.deal;

      if(HistoryDealSelect(dealTicket)) {
         long dealMagic = HistoryDealGetInteger(dealTicket, DEAL_MAGIC);
         if(dealMagic != MagicNumber) return;

         ENUM_DEAL_ENTRY dealEntry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(dealTicket, DEAL_ENTRY);

         if(dealEntry == DEAL_ENTRY_OUT || dealEntry == DEAL_ENTRY_INOUT) {
            double profit = HistoryDealGetDouble(dealTicket, DEAL_PROFIT);
            double balance = AccountInfoDouble(ACCOUNT_BALANCE);

            stats.dailyPL += (profit / balance) * 100;

            if(profit > 0) {
               stats.consecutiveWins++;
               stats.consecutiveLosses = 0;
            } else if(profit < 0) {
               stats.consecutiveLosses++;
               stats.consecutiveWins = 0;
            }

            // Reset position state on full close
            if(!PositionSelect(_Symbol)) {
               ResetPositionState();
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Draw Zones on Chart                                               |
//+------------------------------------------------------------------+
void DrawZones()
{
   ObjectsDeleteAll(0, "SWSI_Zone_");

   for(int i = 0; i < ArraySize(zones); i++) {
      if(zones[i].status == ZONE_MITIGATED) continue;

      string objName = "SWSI_Zone_" + IntegerToString(i);
      datetime time1 = zones[i].formationTime;
      datetime time2 = TimeCurrent() + PeriodSeconds(PERIOD_CURRENT) * 30;

      color zoneColor = zones[i].isBullish ? DemandZoneColor : SupplyZoneColor;

      ObjectCreate(0, objName, OBJ_RECTANGLE, 0, time1, zones[i].zoneHigh, time2, zones[i].zoneLow);
      ObjectSetInteger(0, objName, OBJPROP_COLOR, zoneColor);
      ObjectSetInteger(0, objName, OBJPROP_FILL, true);
      ObjectSetInteger(0, objName, OBJPROP_BACK, true);

      // Zone label
      string labelName = objName + "_label";
      ObjectCreate(0, labelName, OBJ_TEXT, 0, time1, zones[i].zoneHigh);

      string zoneType = (zones[i].type == ZONE_RBD ? "RBD" :
                        (zones[i].type == ZONE_DBR ? "DBR" :
                        (zones[i].type == ZONE_RBR ? "RBR" : "DBD")));

      ObjectSetString(0, labelName, OBJPROP_TEXT,
                      "SWSI " + zoneType + " Q:" + IntegerToString(zones[i].qualityScore));
      ObjectSetInteger(0, labelName, OBJPROP_COLOR, zoneColor);
      ObjectSetInteger(0, labelName, OBJPROP_FONTSIZE, 8);
   }
}

//+------------------------------------------------------------------+
//| Draw Signal on Chart                                              |
//+------------------------------------------------------------------+
void DrawSignal(bool isBuy, double entry, double sl, double tp)
{
   string prefix = "SWSI_Sig_" + IntegerToString(TimeCurrent());

   // Entry arrow
   string arrowName = prefix + "_arrow";
   ObjectCreate(0, arrowName, OBJ_ARROW, 0, TimeCurrent(), entry);
   ObjectSetInteger(0, arrowName, OBJPROP_ARROWCODE, isBuy ? 233 : 234);
   ObjectSetInteger(0, arrowName, OBJPROP_COLOR, isBuy ? clrLime : clrRed);
   ObjectSetInteger(0, arrowName, OBJPROP_WIDTH, 2);

   // SL line
   string slName = prefix + "_sl";
   ObjectCreate(0, slName, OBJ_HLINE, 0, 0, sl);
   ObjectSetInteger(0, slName, OBJPROP_COLOR, clrRed);
   ObjectSetInteger(0, slName, OBJPROP_STYLE, STYLE_DOT);

   // TP line
   string tpName = prefix + "_tp";
   ObjectCreate(0, tpName, OBJ_HLINE, 0, 0, tp);
   ObjectSetInteger(0, tpName, OBJPROP_COLOR, clrLime);
   ObjectSetInteger(0, tpName, OBJPROP_STYLE, STYLE_DOT);
}
//+------------------------------------------------------------------+
