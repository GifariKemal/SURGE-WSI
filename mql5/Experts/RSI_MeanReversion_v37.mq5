//+------------------------------------------------------------------+
//|                                      RSI_MeanReversion_v37.mq5   |
//|                                              SURIOTA Trading Team |
//|                                                                   |
//| RSI Mean Reversion Strategy v3.7                                  |
//| Backtest: +618% return, 37.7% WR, 1.22 PF (2020-2026)            |
//+------------------------------------------------------------------+
#property copyright "SURIOTA Trading Team"
#property link      "https://suriota.com"
#property version   "3.70"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>
#include <Trade\DealInfo.mqh>

//--- Input Parameters
input group "=== RSI Settings ==="
input int      RSI_Period = 10;              // RSI Period
input int      RSI_Oversold = 42;            // RSI Oversold Level
input int      RSI_Overbought = 58;          // RSI Overbought Level

input group "=== ATR Settings ==="
input int      ATR_Period = 14;              // ATR Period
input int      ATR_Lookback = 100;           // ATR Percentile Lookback
input double   ATR_Min_Pct = 20.0;           // Min ATR Percentile
input double   ATR_Max_Pct = 80.0;           // Max ATR Percentile

input group "=== SL/TP Settings ==="
input double   SL_Multiplier = 1.5;          // SL Multiplier (ATR)
input double   TP_Low = 2.4;                 // TP Low Volatility (ATR < 40%)
input double   TP_Med = 3.0;                 // TP Medium Volatility (ATR 40-60%)
input double   TP_High = 3.6;                // TP High Volatility (ATR > 60%)
input double   TP_Time_Bonus = 0.35;         // TP Bonus (12-16 UTC)

input group "=== Time Filter ==="
input int      Trading_Start_Hour = 7;       // Trading Start Hour (UTC)
input int      Trading_End_Hour = 22;        // Trading End Hour (UTC)
input int      Skip_Hour = 12;               // Skip Hour (UTC)
input int      TP_Bonus_Start = 12;          // TP Bonus Start Hour
input int      TP_Bonus_End = 16;            // TP Bonus End Hour

input group "=== Risk Management ==="
input double   Risk_Percent = 1.0;           // Risk Per Trade (%)
input double   Max_Daily_Loss = 3.0;         // Max Daily Loss (%)
input int      Max_Holding_Hours = 46;       // Max Holding Period (Hours)
input int      Magic_Number = 20250131;      // Magic Number

//--- Global Variables
CTrade         trade;
CPositionInfo  posInfo;
CAccountInfo   accInfo;

int            rsiHandle;
int            atrHandle;
double         rsiBuffer[];
double         atrBuffer[];
double         atrHistory[];

datetime       lastBarTime;
datetime       positionOpenTime;
double         dailyStartBalance;
datetime       dailyResetTime;
bool           circuitBreakerTriggered;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   //--- Initialize indicators
   rsiHandle = iRSI(_Symbol, PERIOD_H1, RSI_Period, PRICE_CLOSE);
   atrHandle = iATR(_Symbol, PERIOD_H1, ATR_Period);

   if(rsiHandle == INVALID_HANDLE || atrHandle == INVALID_HANDLE)
   {
      Print("Error creating indicators!");
      return INIT_FAILED;
   }

   //--- Set arrays as series
   ArraySetAsSeries(rsiBuffer, true);
   ArraySetAsSeries(atrBuffer, true);
   ArraySetAsSeries(atrHistory, true);

   //--- Initialize trade object
   trade.SetExpertMagicNumber(Magic_Number);
   trade.SetDeviationInPoints(10);

   //--- Set order filling type based on broker support
   ENUM_ORDER_TYPE_FILLING filling = GetSupportedFilling();
   trade.SetTypeFilling(filling);
   Print("Order filling type: ", EnumToString(filling));

   //--- Initialize daily tracking
   dailyStartBalance = accInfo.Balance();
   dailyResetTime = GetDailyResetTime();
   circuitBreakerTriggered = false;
   lastBarTime = 0;

   Print("RSI Mean Reversion v3.7 initialized");
   Print("Symbol: ", _Symbol, " | Magic: ", Magic_Number);

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(rsiHandle != INVALID_HANDLE) IndicatorRelease(rsiHandle);
   if(atrHandle != INVALID_HANDLE) IndicatorRelease(atrHandle);
   Print("RSI Mean Reversion v3.7 stopped");
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Check for new H1 bar
   datetime currentBarTime = iTime(_Symbol, PERIOD_H1, 0);
   if(currentBarTime == lastBarTime) return;
   lastBarTime = currentBarTime;

   //--- Daily reset check
   CheckDailyReset();

   //--- Circuit breaker check
   if(circuitBreakerTriggered)
   {
      return;
   }

   if(CheckCircuitBreaker())
   {
      circuitBreakerTriggered = true;
      Print("CIRCUIT BREAKER TRIGGERED! Trading paused.");
      return;
   }

   //--- Copy indicator data
   if(CopyBuffer(rsiHandle, 0, 0, 3, rsiBuffer) < 3) return;
   if(CopyBuffer(atrHandle, 0, 0, ATR_Lookback + 1, atrHistory) < ATR_Lookback + 1) return;

   double currentRSI = rsiBuffer[1];  // Use completed bar
   double currentATR = atrHistory[1];
   double atrPercentile = CalculateATRPercentile();

   //--- Manage existing position
   if(HasPosition())
   {
      ManagePosition();
      return;
   }

   //--- Check for new entry
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int currentHour = dt.hour;
   int weekday = dt.day_of_week;

   //--- Weekend filter
   if(weekday == 0 || weekday == 6) return;

   //--- Time filter
   if(currentHour < Trading_Start_Hour || currentHour >= Trading_End_Hour) return;
   if(currentHour == Skip_Hour) return;

   //--- ATR filter
   if(atrPercentile < ATR_Min_Pct || atrPercentile > ATR_Max_Pct) return;

   //--- Spread filter: max 30% of ATR
   double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double maxSpread = currentATR * 0.3;
   if(spread > maxSpread)
   {
      static datetime lastSpreadWarn = 0;
      if(TimeCurrent() - lastSpreadWarn > 3600) // Warn once per hour
      {
         Print("Spread too wide: ", spread, " > ", maxSpread);
         lastSpreadWarn = TimeCurrent();
      }
      return;
   }

   //--- Generate signal
   int signal = 0;
   if(currentRSI < RSI_Oversold) signal = 1;       // BUY
   else if(currentRSI > RSI_Overbought) signal = -1; // SELL

   if(signal == 0) return;

   //--- Calculate SL/TP
   double entryPrice = (signal == 1) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                                      : SymbolInfoDouble(_Symbol, SYMBOL_BID);

   double tpMultiplier = GetTPMultiplier(atrPercentile, currentHour);
   double sl, tp;

   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   if(signal == 1) // BUY
   {
      sl = NormalizeDouble(entryPrice - currentATR * SL_Multiplier, digits);
      tp = NormalizeDouble(entryPrice + currentATR * tpMultiplier, digits);
   }
   else // SELL
   {
      sl = NormalizeDouble(entryPrice + currentATR * SL_Multiplier, digits);
      tp = NormalizeDouble(entryPrice - currentATR * tpMultiplier, digits);
   }

   //--- Calculate lot size
   double lotSize = CalculateLotSize(entryPrice, sl);
   if(lotSize <= 0) return;

   //--- Execute trade
   string comment = StringFormat("RSIv37_%s_RSI%.1f", (signal == 1 ? "B" : "S"), currentRSI);

   bool success;
   if(signal == 1)
      success = trade.Buy(lotSize, _Symbol, entryPrice, sl, tp, comment);
   else
      success = trade.Sell(lotSize, _Symbol, entryPrice, sl, tp, comment);

   if(success)
   {
      positionOpenTime = TimeCurrent();
      Print(StringFormat("%s %.2f lots @ %.5f | SL: %.5f | TP: %.5f | RSI: %.1f | ATR%%: %.0f",
            (signal == 1 ? "BUY" : "SELL"), lotSize, entryPrice, sl, tp, currentRSI, atrPercentile));
   }
   else
   {
      Print("Trade failed: ", trade.ResultRetcode(), " - ", trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
//| Calculate ATR Percentile: % of historical values below current    |
//+------------------------------------------------------------------+
double CalculateATRPercentile()
{
   if(ArraySize(atrHistory) < ATR_Lookback + 1) return 50.0;

   double currentATR = atrHistory[1];  // Current bar's ATR
   int countBelow = 0;
   int historyCount = ATR_Lookback - 1;  // Exclude current value

   // Compare against historical values only (skip current at index 1)
   for(int i = 2; i <= ATR_Lookback; i++)
   {
      if(atrHistory[i] < currentATR) countBelow++;
   }

   return historyCount > 0 ? (double)countBelow / historyCount * 100.0 : 50.0;
}

//+------------------------------------------------------------------+
//| Get TP Multiplier based on volatility and time                    |
//+------------------------------------------------------------------+
double GetTPMultiplier(double atrPct, int hour)
{
   double baseTP;

   if(atrPct < 40) baseTP = TP_Low;
   else if(atrPct > 60) baseTP = TP_High;
   else baseTP = TP_Med;

   //--- Add time bonus during London/NY overlap
   if(hour >= TP_Bonus_Start && hour < TP_Bonus_End)
      return baseTP + TP_Time_Bonus;

   return baseTP;
}

//+------------------------------------------------------------------+
//| Get supported order filling type                                   |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING GetSupportedFilling()
{
   uint filling_modes = (uint)SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);

   if((filling_modes & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)
      return ORDER_FILLING_FOK;

   if((filling_modes & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC)
      return ORDER_FILLING_IOC;

   return ORDER_FILLING_RETURN;
}

//+------------------------------------------------------------------+
//| Calculate Lot Size based on risk                                  |
//+------------------------------------------------------------------+
double CalculateLotSize(double entryPrice, double slPrice)
{
   double balance = accInfo.Balance();
   double riskAmount = balance * Risk_Percent / 100.0;
   double slDistance = MathAbs(entryPrice - slPrice);

   if(slDistance <= 0) return 0;

   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   double slTicks = slDistance / tickSize;
   double lotSize = riskAmount / (slTicks * tickValue);

   //--- Round to lot step
   lotSize = MathFloor(lotSize / lotStep) * lotStep;

   //--- Clamp to valid range
   lotSize = MathMax(minLot, MathMin(maxLot, lotSize));

   return NormalizeDouble(lotSize, 2);
}

//+------------------------------------------------------------------+
//| Check if we have an open position                                 |
//+------------------------------------------------------------------+
bool HasPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(posInfo.SelectByIndex(i))
      {
         if(posInfo.Symbol() == _Symbol && posInfo.Magic() == Magic_Number)
            return true;
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Manage existing position                                          |
//+------------------------------------------------------------------+
void ManagePosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(posInfo.SelectByIndex(i))
      {
         if(posInfo.Symbol() == _Symbol && posInfo.Magic() == Magic_Number)
         {
            //--- Check max holding time
            datetime openTime = posInfo.Time();
            int holdingHours = (int)((TimeCurrent() - openTime) / 3600);

            if(holdingHours >= Max_Holding_Hours)
            {
               Print("Max holding time reached (", holdingHours, " hours). Closing position.");
               trade.PositionClose(posInfo.Ticket());
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check Circuit Breaker (daily loss limit)                          |
//+------------------------------------------------------------------+
bool CheckCircuitBreaker()
{
   double currentBalance = accInfo.Balance();
   double dailyLossPct = (dailyStartBalance - currentBalance) / dailyStartBalance * 100.0;

   return (dailyLossPct >= Max_Daily_Loss);
}

//+------------------------------------------------------------------+
//| Check and perform daily reset                                     |
//+------------------------------------------------------------------+
void CheckDailyReset()
{
   datetime now = TimeCurrent();

   if(now >= dailyResetTime + 86400) // 24 hours
   {
      dailyStartBalance = accInfo.Balance();
      dailyResetTime = GetDailyResetTime();
      circuitBreakerTriggered = false;
      Print("Daily reset. Balance: $", DoubleToString(dailyStartBalance, 2));
   }
}

//+------------------------------------------------------------------+
//| Get daily reset time (midnight UTC)                               |
//+------------------------------------------------------------------+
datetime GetDailyResetTime()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   dt.hour = 0;
   dt.min = 0;
   dt.sec = 0;
   return StructToTime(dt);
}

//+------------------------------------------------------------------+
//| Trade transaction handler                                         |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
{
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
   {
      if(trans.deal_type == DEAL_TYPE_BUY || trans.deal_type == DEAL_TYPE_SELL)
      {
         // Position opened or closed
         CDealInfo deal;
         if(deal.SelectByIndex(HistoryDealsTotal() - 1))
         {
            if(deal.Magic() == Magic_Number && deal.Entry() == DEAL_ENTRY_OUT)
            {
               double profit = deal.Profit() + deal.Swap() + deal.Commission();
               Print(StringFormat("Position closed. P/L: $%.2f", profit));
            }
         }
      }
   }
}
//+------------------------------------------------------------------+
