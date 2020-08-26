
class RunMode:
    simulate = "simulate"
    demo = "demo"
    real_time = "real_time"


class OrderStatus:
    pending = "pending"
    running = "running"
    complete = "complete"
    cancelled = "cancelled"


class OrderType:
    limit = "LMT"
    limit_on_close = "LOC"
    stop = "STP"


class PredictionType:
    rise = "rise"
    high = "high"
    low = "low"
