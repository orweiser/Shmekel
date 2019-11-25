using System;

namespace TradeManager
{

    public class Candle
    {
        public DateTime DateTime;
        public double Open;
        public double High;
        public double Low;
        public double Close;
        public double Volume;

        public double PredictUp;
    
        public double OpenToHighRatio;
        public double OpenToLowRatio;
        public double OpenToCloseRatio;
        public double PreviousClose;
        public double PreviousCloseToOpenRatio;


        public void print()
        {
            Console.WriteLine("datetime: "+this.DateTime);
            Console.WriteLine("open: " + this.Open);
            Console.WriteLine("high: " + this.High);
            Console.WriteLine("low: " + this.Low);
            Console.WriteLine("close: " + this.Close);


            Console.WriteLine("open to high: " + this.OpenToHighRatio);
            Console.WriteLine("open to low: " + this.OpenToLowRatio);
            Console.WriteLine("open to close: " + this.OpenToCloseRatio);
            Console.WriteLine("previous close: " + this.PreviousClose);

            Console.WriteLine("previous close to open: " + this.PreviousCloseToOpenRatio);
        }

    }
}