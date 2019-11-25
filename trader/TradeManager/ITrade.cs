using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace TradeManager
{
    interface ITrade
    {
        void Offline();
    }

    public enum TradeStatus { Pending,Active,ExitNoEntry,ExitTakeProfit,ExitStopLoss,ExitOnClose,ExitOnTimeGap}

    public class SimpleTrade:ITrade
    {
        public TimeSpan TimeGapThreshold;


        public TradeStatus Status;
        public int EntryCandleNumber;
        public int ExitCandleNumber;

        public double PL;
        public double Shares = 1000;
        public double CostPerShare = 0.005;
        public double MinCost = 1.5;
        public double EntryPrice;
        public double StopLossPrice;
        public double TakeProfitPrice;
        
        public List<Candle> TradeCandles;

        public SimpleTrade(TimeSpan timeGapThreshold)
        {
            TimeGapThreshold = timeGapThreshold;
        }

        public void Offline()
        {
            try
            {
                this.Status = TradeStatus.Pending;
                
                for (int i = 0; i < TradeCandles.Count; i++)
                {
                    if (Status == TradeStatus.Pending)
                    {
                        if (TradeCandles[i].Low < EntryPrice && EntryPrice < TradeCandles[i].High)
                        {
                            Status = TradeStatus.Active;
                            EntryCandleNumber = i;
                        }
                    }

                    if (Status == TradeStatus.Active)
                    {
                        //**** STOP LOSS ****//
                        if (StopLossPrice > TradeCandles[i].Low)
                        {
                            Status = TradeStatus.ExitStopLoss;
                            ExitCandleNumber = i;
                            CalculatePL(this.StopLossPrice);
                            break;
                        }

                        //**** TAKE PROFIT ****//
                        if (TakeProfitPrice < TradeCandles[i].High)
                        {
                            Status = TradeStatus.ExitTakeProfit;
                            ExitCandleNumber = i;
                            CalculatePL(this.TakeProfitPrice);
                            break;
                        }

                        //**** EXIT ON LAST CANDLE ****//
                        if (i == TradeCandles.Count - 1)
                        {
                            Status = TradeStatus.ExitOnClose;
                            ExitCandleNumber = i;
                            CalculatePL(TradeCandles[i].Close);
                            break;
                        }

                        //**** EXIT ON TIME GAP AHEAD ****//
                        var nexTimeSpan = TradeCandles[i + 1].DateTime - TradeCandles[i].DateTime;
                        if (nexTimeSpan > TimeGapThreshold)
                        {
                            Status = TradeStatus.ExitOnTimeGap;
                            ExitCandleNumber = i;
                            CalculatePL(TradeCandles[i].Close);
                            break;
                            
                        }

                    }
                        
                }

                if (Status == TradeStatus.Pending){Status = TradeStatus.ExitNoEntry;}
                
                
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
                throw;
            }
        }

        private void CalculatePL(double exitPrice)
        {
            try
            {

                double rawPerShare = exitPrice - EntryPrice;
                double raw = Shares*rawPerShare;
                double cost = Shares * CostPerShare;
                if (cost < MinCost){cost = MinCost;}

                this.PL = raw - cost;
                
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
                throw;
            }


        }
    }
}