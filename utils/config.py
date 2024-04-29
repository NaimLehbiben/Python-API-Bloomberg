from datetime import datetime

TICKER = "RIY INDEX"
START_DATE = datetime(2000,1,31).date()
END_DATE =  datetime(2024,2,29).date()
USE_PICKLE_UNIVERSE = True
CURRENCY ="USD"
STEP_VOL = -6
STEP_SLOPE = -1
RISK_FREE_RATE_TICKER = "US0003M Index"
SLOPE_ALPHA = 0.02
WEGHTS_TYPE = "Equally Weighted"