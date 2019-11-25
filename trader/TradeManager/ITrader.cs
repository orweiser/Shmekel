using System;
using System.Collections.Generic;
using System.Linq;
using Globals;

namespace TradeManager
{
    interface ITrader
    {
        void Learn(List<Candle> iCandles,int resolution);
        
        void OfflineTrade(List<Candle> offlineCandles);
    }

    public class UpSingleTrader : ITrader
    {
        //strategy type 
        public StrategyType StrategyType = StrategyType.Basic;

        public Strategy CurrentStrategy;
        public List<Strategy> Strategies=new List<Strategy>();
        public List<Strategy> SelectedStrategies = new List<Strategy>();

        public TimeSpan TimeGapThreshold;
        public double InitialFunds;

        public double Funds;
        public double PreviousCloseToEntryRatio;
        public double OpenToTakeProfitRatio;
        public double OpenToStopLossRatio;

        /// <summary>
        /// basic trader - for candle up trading
        /// predictor for this trader: 0 to 1 predicting if candle will close higher then open
        /// </summary>
        /// <param name="timeGapThreshold">TimeSpan type, defines the threshold for time gap (exit on close) </param>
        /// <param name="initialFunds">initial funds for trader</param>
        public UpSingleTrader(TimeSpan timeGapThreshold,double initialFunds = 0)
        {
            TimeGapThreshold = timeGapThreshold;
            InitialFunds = initialFunds;
        }

        
        /// <summary>
        /// Learn the best strategy for a given set of candles
        /// </summary>
        /// <param name="iCandles">list of candles to learn</param>
        /// <param name="resolution">Learn resolution - used for threshold, combinations </param>
        public void Learn(List<Candle> iCandles,int resolution)
        {
            GenerateStrategies(resolution);
            LearnStrategies(iCandles);

            //Generate all strategy combinations
            void GenerateStrategies(int slices)
            {
                List<double> predictorThresholds = new List<double>();
                List<double> o2HValues = new List<double>();
                List<double> o2LValues = new List<double>();
                List<double> p2OValues = new List<double>();

                var upCandles = iCandles.Where(x => x.Close > x.Open);

                //get average ratios
                var avgO2H = upCandles.Average(p => p.OpenToHighRatio);
                var avgO2L = upCandles.Average(p => p.OpenToLowRatio);
                var avgP2O = upCandles.Average(p => p.PreviousCloseToOpenRatio);

                //**** set limits and fill arrays ****

                //Thresholds
                switch (StrategyType)
                {
                    case StrategyType.Basic:
                        double thresholdStep = Convert.ToDouble(1) / slices;
                        for (int i = 0; i < slices; i++)
                        {
                            predictorThresholds.Add(i*thresholdStep);
                            //Console.WriteLine();
                        }

                        predictorThresholds.Remove(1);
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }

                //OPEN TO HIGH
                double maxO2H = avgO2H * 2;
                double minO2H = avgO2H / 2;
                double spanO2H = maxO2H - minO2H;
                double stepO2H = spanO2H / slices;
                for (int i = 0; i <= slices; i++){o2HValues.Add(minO2H+i*stepO2H);}
               
                //OPEN TO LOW
                double maxO2L = avgO2L / 2;
                double minO2L = avgO2L * 2;
                double spanO2L = Math.Abs(maxO2L - minO2L);
                double stepO2L = spanO2L / slices;
                for (int i = 0; i <= slices; i++){o2LValues.Add(minO2L + i * stepO2L);}

                //PREVIOUS TO OPEN
                double maxP2O = avgP2O * 2;
                double minP2O = avgP2O / 2;
                double spanP2O = Math.Abs(maxP2O - minP2O);
                double stepP2O = spanP2O / slices;
                for (int i = 0; i <= slices; i++) { p2OValues.Add(minP2O + i * stepP2O); }

                

                for (int i = 0; i < o2HValues.Count; i++)
                {
                    for (int j = 0; j < o2LValues.Count; j++)
                    {
                        for (int k = 0; k < p2OValues.Count; k++)
                        {
                            foreach (double predictorThreshold in predictorThresholds)
                            {

                                Dictionary<StrategyRatioType, double> ratios = new Dictionary<StrategyRatioType, double>();
                                ratios[StrategyRatioType.Entry] = p2OValues[k];
                                ratios[StrategyRatioType.StopLoss] = o2LValues[j];
                                ratios[StrategyRatioType.TakeProfit] = o2HValues[i];
                                ratios[StrategyRatioType.PredictorThreshold] = predictorThreshold;

                                Strategy strategy = new Strategy(StrategyType.Basic, ratios);
                                strategy.Score.TotalFunds = InitialFunds;
                                strategy.Score.InitialFunds = InitialFunds;
                                Strategies.Add(strategy);
                               
                            }

                            
                        }

                    }
                }
               
            }
            
            //learn all strategies by offline trade
            void LearnStrategies(List<Candle> candles)
            {
                foreach (Strategy strategy in Strategies)
                {
                    CurrentStrategy = strategy;
                    OfflineTrade(candles);
                    CurrentStrategy.UpdateScore();
                    
                    if (CurrentStrategy.Score.TotalFunds>InitialFunds)
                    {
                        SelectedStrategies.Add(CurrentStrategy);
                        Logger.AddEntry("Strategy added to selected, total funds: " + CurrentStrategy.Score.TotalFunds, LogEntryType.Info);

                    }
                    else
                    {
                        Logger.AddEntry("Strategy failed, total funds: "+CurrentStrategy.Score.TotalFunds,LogEntryType.Info);
                    }
                    
                    //Console.WriteLine(strategy.ToCsvLine());
                    // Console.ReadKey();
                }
            }
            
        }




        public void OfflineTrade(List<Candle> offlineCandles)
        {
            //grace period - minimum candles before checking if strategy is a loosing strategy 
            double gracePeriodCandle = Math.Round(offlineCandles.Count*InputVariables.GlobalConfig.GracePeriodRatio);

            for (int i = 0; i < offlineCandles.Count-CurrentStrategy.MaxCandlesPerTrade; i++)
            {
                //check predict up for trade open
                CurrentStrategy.Score.TotalCandleCount++;
                if (offlineCandles[i].PredictUp <= CurrentStrategy.Ratios[StrategyRatioType.PredictorThreshold])
                {
                    continue;
                }

                //Check for grace period status
                if (i >gracePeriodCandle)
                {
                    if (CurrentStrategy.Score.TotalFunds<InitialFunds)
                    {
                        //Console.WriteLine();
                        Logger.AddEntry("NEGATIVE FUNDS AFTER GRACE PERIOD, EXIT OFFLINE TRADE", LogEntryType.Info);
                        break;
                    }
                    

                }
                
                //open trade
                SimpleTrade trade = new SimpleTrade(TimeGapThreshold);
                trade.Shares = 1000;
                trade.TradeCandles = offlineCandles.GetRange(i, CurrentStrategy.MaxCandlesPerTrade);
                trade.EntryPrice = offlineCandles[i].PreviousClose * (1 + CurrentStrategy.Ratios[StrategyRatioType.Entry]);
                trade.TakeProfitPrice = offlineCandles[i].PreviousClose * (1 + CurrentStrategy.Ratios[StrategyRatioType.TakeProfit]);
                trade.StopLossPrice = offlineCandles[i].PreviousClose * (1 + CurrentStrategy.Ratios[StrategyRatioType.StopLoss]);

                //run offline
                trade.Offline();
                
                //update strategy
                CurrentStrategy.UpdateTrade(trade.PL,trade.Status);

                //output score
                if (InputVariables.GlobalConfig.OutputLevel == OutputLevel.Trade)
                {
                    Console.WriteLine(i + " | {0} , PL: {1}", trade.Status, trade.PL);
                    Console.WriteLine("funds: {0} ", CurrentStrategy.Score.TotalFunds);
                }

            }

        }

        

    }

}