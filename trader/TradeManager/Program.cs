using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Globals;

namespace TradeManager
{
    class Program
    {
        static void Main(string[] args)
        {
            //DONE: send csv format to dror/yishay

            //DONE: ADD prediction THRESHOLD FOR 0/1 PREDICT to learn function
            //todo: input path , output path , config path

            //todo: ARGUMENTS: FOR EACH STRATEGY OUTPUT CONFIGURABLE PARAMETERS
            //todo: add exit on market if time gap
            ////todo: output type w/o details
            //todo: flags: train / validation 
            //todo: train output: table with all configurable params and scores
            //todo: validation: output all data

            string input = String.Join(" ",args);
            input = "-runmode = train " + "-input = D:\\shmekels\\downloads\\learn.csv " + "-output = D:\\shmekels\\downloads\\aapl_out.csv "+ "-config = D:\\shmekels\\downloads\\aapl_config.xml";
           // input = "-runmode = train " + "-input = D:\\shmekels\\downloads\\learn.csv " + "-output = D:\\shmekels\\downloads\\aapl_out.csv " + "-config = D:\\shmekels\\downloads\\aapl_config.xml";
            
            try
            {
                ParseInput();
                Run();
                
            }
            catch (Exception e)
            {

                Console.WriteLine("Main Run error");
                Console.WriteLine(e);
                Logger.CriticalLogList.Add("!!! EXIT ON MAIN RUN ERROR: "+e);
                
            }
            

            Logger.Save();
            Logger.SummaryToConsole();


            void ParseInput()
            {
                try
                {
                    InputVariables.ParseInput(input);
                }
                catch (Exception e)
                {
                    Console.WriteLine(e);
                    throw new Exception("InputVariables --> ParseInput exception: \r\n " + e);
                }
            }
            
            void Run()
            {
                Logger.AddEntry("START RUN FUNCTION ", LogEntryType.Info);

                //Initialize and load asset
                Asset asset = new Asset();
                asset.LoadFromCsv(InputVariables.InputPath);

                switch (InputVariables.RunMode)
                {
                    case RunMode.Train:
                        asset.Train();
                        break;
                    case RunMode.Validate:
                        break;
                    case RunMode.Offline:
                        break;
                    case RunMode.Online:
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }


                Logger.AddEntry("FINISH RUN FUNCTION", LogEntryType.Info);

            }
        }
    }
}
