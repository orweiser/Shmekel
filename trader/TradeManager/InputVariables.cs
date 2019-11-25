using System;
using System.Collections.Generic;
using System.IO;
using TradeManager;
using XmlFunctions;

namespace Globals
{
    public enum RunMode { Train,Validate,Offline,Online }
    public enum OutputLevel { Summary, Asset, Trade }
    public static class InputVariables
    {
        public static RunMode RunMode = RunMode.Train;
        public static string InputPath;
        public static string OutputPath;
        public static Config GlobalConfig;

        public static void ParseInput(string inputStr)
        {
            string inputSplitter = "-";
            string inputDictSplitter = "=";
            Dictionary<string,string> input = new Dictionary<string, string>();

            ParseInputToDictionary();
            UpdateRunMode();
            UpdateGlobalConfig();
            UpdateInputPath();
            UpdateOutputPath();

            

            //**** FUNCTIONS ****
            void ParseInputToDictionary()
            {
                    Logger.AddEntry("PARSING INPUT TO DICTIONARY",LogEntryType.Info);
                    if (inputStr.Length == 0){throw new Exception("No Arguments given");}
                    Logger.AddEntry($"INPUT STRING -->  '{inputStr}'",LogEntryType.Info);
                    string[] split = inputStr.Split(new[] { inputSplitter }, StringSplitOptions.RemoveEmptyEntries);
                    foreach (string s in split)
                    {
                        string[] dictSplit = s.Split(new[] { inputDictSplitter }, StringSplitOptions.RemoveEmptyEntries);
                        string key = dictSplit[0].Trim().ToLower();
                        string value = dictSplit[1].Trim().ToLower();
                        input[key] = value;
                        Logger.AddEntry($"KEY --> '{key}' , VALUE --> '{value}'",LogEntryType.Info);
                    }
                    
               
            }

            void UpdateRunMode()
            {
                Logger.AddEntry("UPDATE RUN MODE",LogEntryType.Info);
                if (!input.ContainsKey("runmode"))
                {
                    string errorMsg = "UpdateRunMode Error : No runmode argument was given: ";
                    Logger.AddEntry(errorMsg , LogEntryType.Critical);
                    throw new Exception(errorMsg);
                }
                if (input["runmode"] == RunMode.Train.ToString().ToLower())
                {
                    RunMode = RunMode.Train;
                }
                else if (input["runmode"] == RunMode.Offline.ToString().ToLower())
                {
                    RunMode = RunMode.Offline;
                }
                else if (input["runmode"] == RunMode.Validate.ToString().ToLower())
                {
                    RunMode = RunMode.Validate;
                }
                else
                {
                    string errorMsg = String.Format("UpdateRunMode Error : invalid runmode input:{0} ", input["runmode"]);
                    Logger.AddEntry(errorMsg,LogEntryType.Critical);
                    throw new ArgumentException(errorMsg);
                }
                
            }

            void UpdateGlobalConfig()
            {
                if (input.ContainsKey("config"))
                {
                    string path = input["config"];
                    try
                    {
                        using (XmlStream stream = new XmlStream())
                        {
                            Logger.AddEntry("LOADING CONFIG FILE --> " + path, LogEntryType.Info);
                            GlobalConfig = stream.ReadFromXmlFile<Config>(path);
                        }
                    }
                    catch (Exception e)
                    {
                        string errorMsg = "UpdateGlobalConfig load Error: " + e;
                        Logger.AddEntry(errorMsg, LogEntryType.Critical);
                        throw new ArgumentException(errorMsg);
                    }
                    

                    
                }
                else
                {
                 
                    Logger.AddEntry("NO CONFIG FILE INPUT, LOADING DEFAULT CONFIG ", LogEntryType.Info);
                    GlobalConfig = new Config();
                   
                }

            }

            void UpdateInputPath()
            {
                if (!input.ContainsKey("input"))
                {
                    string errorMsg = "UpdateInputPath Error : No input argument was given: ";
                    Logger.AddEntry(errorMsg, LogEntryType.Critical);
                    throw new Exception(errorMsg);
                }

                string path = input["input"];
                var content = File.ReadAllLines(path);
               
                if (content.Length == 0)
                {
                    string msg = "INPUT FILE IS EMPTY --> " + path;
                    Logger.AddEntry(msg,LogEntryType.Critical);
                    throw new Exception(msg);
                }

                Logger.AddEntry("SETTING INPUT PATH --> " + path, LogEntryType.Info);

                InputPath = path;
            }

            void UpdateOutputPath()
            {
                if (!input.ContainsKey("output"))
                {
                    string errorMsg = "UpdateOutputPath Error : No output argument was given: ";
                    Logger.AddEntry(errorMsg, LogEntryType.Critical);
                    throw new Exception(errorMsg);
                }

                string path = input["output"];
                Logger.AddEntry("SETTING OUTPUT PATH --> " + path, LogEntryType.Info);
                
                OutputPath = path;
            }
        }

    }

    public class Config
    {
        public double MinProfitRatio = 0; //default value = 0
        public double GracePeriodRatio = 0.2;//default value 0.2
        public int TrainResolution = 10; //default value = 10
        public int MaxCandlesPerTrade = 10; //default value = 10
        public StrategyType Type = StrategyType.Basic; 
        public OutputLevel OutputLevel = OutputLevel.Summary; //default value


        //csv stracture
        public int DateTimeIndex = 0; //default value = 0
        public int OpenIndex = 1;//default value = 1
        public int HighIndex = 2;//default value = 2
        public int LowIndex = 3;//default value = 3
        public int CloseIndex = 4;//default value = 4
        public int VolumeIndex = 5;//default value = 5

        public int PredictUpIndex = 6;//default value = 6

    }

    
}