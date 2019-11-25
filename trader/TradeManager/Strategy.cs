using System;
using System.Collections.Generic;
using Globals;

namespace Globals
{
}

namespace TradeManager
{
    public enum StrategyType {Basic }
    public enum StrategyRatioType { Entry,TakeProfit,StopLoss,PredictorThreshold}

    public class StrategyScore
    {
        public int TotalTradesCount;
        public int TotalCandleCount;

        public double InitialFunds; //initial funds in strategy
        public double TotalFunds; //total funds at the end of trade period
        public double MaxFunds;//max funds during trade period
        public double MinRequiredFunds;//max drop in funds during trade period (minimum required funds)

        public int ExitOnTakeProfitCount;//number of trades that ended with TakeProfit
        public int ExitOnstopLossCount;//number of trades that ended with StopLoss
        public int ExitOnCloseCount;//number of trades that ended with ExitOnClose
        public int ExitOnNoEntryCount;//number of trades that did not enter 

        public double TotalToMinRatio;
        public double MaxToMinRatio;
        public double MissedEntryRatio;

        public string ToCsvLine()
        {
            return String.Format("{0},{1},{2}," +
                                 "{3},{4},{5}," +
                                 "{6},{7},{8}," +
                                 "{9},{10}",
                InitialFunds,TotalFunds,MaxFunds,
                MinRequiredFunds,TotalToMinRatio,MaxToMinRatio,
                MissedEntryRatio,ExitOnTakeProfitCount,ExitOnstopLossCount,
                ExitOnCloseCount,ExitOnNoEntryCount);
        }

        public string ToCsvHeader()
        {
            return "InitialFunds,TotalFunds,MaxFunds,MinRequiredFunds," +
                   "TotalToMinRatio,MaxToMinRatio,MissedEntryRatio," +
                   "ExitOnTakeProfitCount,ExitOnstopLossCount,ExitOnCloseCount,ExitOnNoEntryCount";
        }

    }
    public class Strategy
    {
        public StrategyType Type;
        public Dictionary<StrategyRatioType, double> Ratios;
        public int MaxCandlesPerTrade=InputVariables.GlobalConfig.MaxCandlesPerTrade;

        public StrategyScore Score = new StrategyScore();


        public Strategy(StrategyType type, Dictionary<StrategyRatioType, double> ratios)
        {
            this.Type = type;
            this.Ratios = ratios;
        }

        public string ToCsvLine()
        {
            string ratios = "";
            foreach (KeyValuePair<StrategyRatioType, double> keyValuePair in Ratios)
            {
                ratios += keyValuePair.Value + ",";
            }
            return String.Format("{0},{1}{2},{3}",
                Type,ratios,MaxCandlesPerTrade,Score.ToCsvLine());
        }

        public string ToCsvHeader()
        {
            string ratios = "";
            foreach (KeyValuePair<StrategyRatioType, double> keyValuePair in Ratios)
            {
                ratios += keyValuePair.Key + "Ratio,";
            }
            return String.Format("Type,{0}MaxCandlesPerTrade,{1}",ratios,Score.ToCsvHeader());
        }

        public void UpdateTrade(double tradePl,TradeStatus tradeStatus)
        {
            Score.TotalTradesCount++;
            Score.TotalFunds += tradePl;
            if (Score.TotalFunds > Score.MaxFunds)
            {
                Score.MaxFunds = Score.TotalFunds;
            }
            else if (Score.MaxFunds-Score.TotalFunds>Score.MinRequiredFunds)
            {
                Score.MinRequiredFunds = Score.MaxFunds - Score.TotalFunds;
            }

            switch (tradeStatus)
            {
                case TradeStatus.ExitNoEntry:
                    Score.ExitOnNoEntryCount++;
                    break;
                case TradeStatus.ExitTakeProfit:
                    Score.ExitOnTakeProfitCount++;
                    break;
                case TradeStatus.ExitStopLoss:
                    Score.ExitOnstopLossCount++;
                    break;
                case TradeStatus.ExitOnClose:
                    Score.ExitOnCloseCount++;
                    break;
            }
        }

        public void UpdateScore()
        {
            Score.MaxToMinRatio = Score.MaxFunds / Score.MinRequiredFunds;
            Score.TotalToMinRatio = Score.TotalFunds / Score.MinRequiredFunds;
            Score.MissedEntryRatio = Convert.ToDouble(Score.ExitOnNoEntryCount)  / Score.TotalTradesCount;
        }

        public void PrintOut()
        {
            Console.WriteLine("\r\nSUMMARY:\r\n");
            Console.WriteLine("predictor threshold: " +Ratios[StrategyRatioType.PredictorThreshold]);

            Console.WriteLine("stoplosses: " + Score.ExitOnstopLossCount);
            Console.WriteLine("takeprofits: " + Score.ExitOnTakeProfitCount);
            Console.WriteLine("noentry: " + Score.ExitOnNoEntryCount);
            Console.WriteLine("onclose: " + Score.ExitOnCloseCount);
            Console.WriteLine("candles per trade: " + MaxCandlesPerTrade);

            Console.WriteLine("Total funds: " + Score.TotalFunds);
            Console.WriteLine("Max funds: " + Score.MaxFunds);
            Console.WriteLine("Min funds: " + Score.MinRequiredFunds);
            Console.WriteLine("Max/min ratio: " + Score.MaxToMinRatio);
            Console.WriteLine("Total/min ratio: " + Score.TotalToMinRatio);

        }


    }

    
    
}