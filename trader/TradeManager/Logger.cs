using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Mime;
using System.Security.Cryptography.X509Certificates;
using System.Threading;

namespace Globals
{
    public enum LogEntryType { Info,Error,Critical }
    public static class Logger
    {
        public static DateTime StartDateTime = DateTime.Now;
        public static List<string> FullLog = new List<string>();
        public static List<string> ErrorsLog = new List<string>();
        public static List<string> CriticalLogList = new List<string>();
        public static List<string> InfoLog = new List<string>();

        public static void AddEntry(string message, LogEntryType type)
        {
            string logMessage = string.Format("{0} | {1} | {2}", DateTime.Now.ToString("G"), type, message);

            //print to console
            if (type == LogEntryType.Info)
            {
                Console.WriteLine(logMessage);
            }
            

            //add to full log
            FullLog.Add(logMessage);

            //add by type
            switch (type)
            {
                case LogEntryType.Info:
                    InfoLog.Add(logMessage);
                    break;
                case LogEntryType.Error:
                    ErrorsLog.Add(logMessage);
                    break;
                case LogEntryType.Critical:
                    CriticalLogList.Add(logMessage);
                    throw new Exception("!!!! CRITICAL EXCEPTION !!!!"+message);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(type), type, null);
            }
        }

        public static void Save()
        {
            string location = AppDomain.CurrentDomain.BaseDirectory;
            string LogsFolder = location + "\\logs";
            if (!Directory.Exists(LogsFolder)){Directory.CreateDirectory(LogsFolder);}

            foreach (string file in Directory.GetFiles(LogsFolder)){File.Delete(file);}

            File.WriteAllLines(LogsFolder+"\\full.log",FullLog);
            if (FullLog.Count > InfoLog.Count){File.WriteAllLines(LogsFolder + "\\info.log", InfoLog);}
            if (ErrorsLog.Count>0){File.WriteAllLines(LogsFolder + "\\errors.log", ErrorsLog);}
            if (CriticalLogList.Count>0){File.WriteAllLines(LogsFolder + "\\critical.log", CriticalLogList);}
           
        }

        public static void SummaryToConsole()
        {
            Console.WriteLine("\r\n\rn\r\n");
            Console.WriteLine("***** SUMMARY *****");
            Console.WriteLine("--------------------------------------------------------");
            Console.WriteLine("Start: "+StartDateTime.ToString("g"));
            TimeSpan duration = DateTime.Now - StartDateTime;
            Console.WriteLine("Duration: "+duration);
            if (ErrorsLog.Count>0)
            {
                Console.WriteLine("Errors found: " + ErrorsLog.Count);
            }

            if (CriticalLogList.Count >0)
            {
                Console.WriteLine("Critical errors found: " + CriticalLogList.Count);
            }
            
           
        }

    }
}