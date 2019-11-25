using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Globals;

namespace TradeManager
{
    /// <summary>
    /// Asset to predict and trade ( can be a stock or ETF )
    /// </summary>
    public class Asset
    {
        public double PredictUpThreshold = 0;// threshold for predictor classification as 1 / 0
        public string CsvPath;//csv path of asset candle set with predictor column
        public List<Candle> Candles = new List<Candle>();//main candle list 

        public TimeSpan TimeGapThreshold = new TimeSpan(1);//minimum time span to indicate a time gap between candles that requires exit on close
        
        /// <summary>
        /// Load Candles from csv to main candle list
        /// </summary>
        /// <param name="csvPath"></param>
        public void LoadFromCsv(string csvPath)
        {
            this.CsvPath = csvPath;

            //update csv indexes from input config
            int dateTimeIndex = InputVariables.GlobalConfig.DateTimeIndex;
            int openIndex = InputVariables.GlobalConfig.OpenIndex;
            int highIndex = InputVariables.GlobalConfig.HighIndex;
            int lowIndex = InputVariables.GlobalConfig.LowIndex;
            int closeIndex = InputVariables.GlobalConfig.CloseIndex;
            int volumeIndex = InputVariables.GlobalConfig.VolumeIndex;

            int predictUpIndex = InputVariables.GlobalConfig.PredictUpIndex;

            double previousClose = 0;

            //load csv
            var lines = File.ReadAllLines(csvPath);
            for (int i = 1; i < lines.Length; i++)
            {
                string[] split = lines[i].Split(',');
                Candle candle = new Candle();

                //raw data
                candle.DateTime = Convert.ToDateTime(split[dateTimeIndex]);
                candle.Open = Convert.ToDouble(split[openIndex]);
                candle.High = Convert.ToDouble(split[highIndex]);
                candle.Low = Convert.ToDouble(split[lowIndex]);
                candle.Close = Convert.ToDouble(split[closeIndex]);
                candle.Volume = Convert.ToDouble(split[volumeIndex]);

                //prediction
                candle.PredictUp = Convert.ToDouble(split[predictUpIndex]);
                
                //ratios
                candle.OpenToCloseRatio = (candle.Close - candle.Open) / candle.Open;
                candle.OpenToHighRatio = (candle.High - candle.Open) / candle.Open;
                candle.OpenToLowRatio = (candle.Low - candle.Open) / candle.Open;
                candle.PreviousClose = previousClose;
                candle.PreviousCloseToOpenRatio = (candle.Open-previousClose)/previousClose;
                
                Candles.Add(candle);
                //candle.print();
                //Console.ReadKey();
                previousClose = candle.Close;
            }

            //set TimeGap
            SetTimeGapThreshold();
            
        }

        /// <summary>
        /// Train the asset trader:
        /// 1 - initialize trader
        /// 2 - run Learn function
        /// 3 - save results 
        /// </summary>
        public void Train()
        {
            Logger.AddEntry("START TRAINING ASSET ",LogEntryType.Info);
            UpSingleTrader trader = new UpSingleTrader(this.TimeGapThreshold,1000);
            trader.Learn(this.Candles,InputVariables.GlobalConfig.TrainResolution);
            Logger.AddEntry("FINISH TRAINING ASSET ", LogEntryType.Info);


            testSave();

            void testSave()
            {

                List<string> lines = new List<string>();
                lines.Add(trader.CurrentStrategy.ToCsvHeader());
                
                foreach (Strategy selectedStrategy in trader.SelectedStrategies)
                {
                    lines.Add(selectedStrategy.ToCsvLine());
                }
             
                File.WriteAllLines(InputVariables.OutputPath,lines);

            }

        }

        /// <summary>
        /// Set the time gap threshold for exit on close
        /// </summary>
        public void SetTimeGapThreshold()
        {
            try
            {
                List<DateTime> times = Candles.Select(x => x.DateTime).ToList();
                Dictionary<TimeSpan, int> tsDictionary = new Dictionary<TimeSpan, int>();
                for (int i = 1; i < times.Count; i++)
                {
                    TimeSpan span = times[i] - times[i - 1];

                    if (tsDictionary.ContainsKey(span))
                    {
                        tsDictionary[span]++;
                    }
                    else
                    {
                        tsDictionary[span] = 1;
                    }
                }

                int max = tsDictionary.Values.Max();
                this.TimeGapThreshold = tsDictionary.Where(x => x.Value == max).Select(x => x.Key).First(); ;
            }
            catch (Exception e)
            {
                Logger.AddEntry("SET TimeGapThreshold FAILURE: "+e.Message,LogEntryType.Critical);
            }
           
            
        }

    }

}