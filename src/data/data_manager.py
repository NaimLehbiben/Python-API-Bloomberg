import pickle

from data_loader import BLP
from datetime import datetime
from utils.utilities import Utilities
from data_loader import BLP

class DataManager:
    
    @staticmethod
    def get_historical_compositions(start_date : datetime, end_date : datetime, ticker : str):
        strFields = ["INDX_MWEIGHT_HIST"]  
        blp = BLP()
        
        rebalancing_dates = Utilities.create_rebalancing_calendar(start_date, end_date)
        composition_par_date = {}


        for date in rebalancing_dates:
            str_date = date.strftime('%Y%m%d')
            compo = blp.bds(strSecurity=[ticker], strFields=strFields, strOverrideField="END_DATE_OVERRIDE", strOverrideValue=str_date)
            list_tickers = compo[strFields[0]].index.tolist()
            composition_par_date[date] = [ticker.split(' ')[0] + ' US Equity' for ticker in list_tickers]
            
        return composition_par_date
        
    @staticmethod
    def get_historical_prices(start_date : datetime, end_date : datetime, tickers : list[str], curr : str):
        
        blp = BLP()
        global_market_data = {}
        tickers_a_supp = []
        
        for ticker in tickers:
            try:
                historical_prices = blp.bdh(strSecurity=[ticker], strFields=["PX_LAST"], startdate=start_date, enddate=end_date, curr=curr, fill="NIL_VALUE")
                historical_prices["PX_LAST"]=historical_prices["PX_LAST"].sort_index(ascending=True)
                
                if not historical_prices["PX_LAST"].empty:
                    global_market_data[ticker] = historical_prices["PX_LAST"]
                else:
                    tickers_a_supp.append[ticker]
                    
            except Exception as e:
                print(f"Erreur lors du traitement du ticker {ticker}: {str(e)}")
            continue
        
        return global_market_data,tickers_a_supp
    
    @staticmethod
    def fetch_backtest_data(start_date : datetime, end_date : datetime, ticker : str, curr : str):
        
    
        composition_par_date = DataManager.get_historical_compositions(start_date, end_date, ticker)
    
        tickers_uniques = list({ticker for composition in composition_par_date.values() for ticker in composition})
        start_date = Utilities.get_rebalancing_date(start_date, step=-6)   
       
        global_market_data, tickers_a_supp = DataManager.get_historical_prices(start_date, end_date, tickers_uniques, curr)
        
        # suppression des tickers sans données 
        composition_par_date = {date: [ticker for ticker in tickers if ticker not in tickers_a_supp] for date, tickers in composition_par_date.items()}    
        
        return composition_par_date, global_market_data     


    @staticmethod
    def fetch_other_US_data(start_date : datetime, end_date : datetime, ticker : str):
                  
         tickers = ["USRINDEX Index","US0003M Index"]
         tickers.append(ticker)
         other_US_data ={}
         
         other_US_data, tickers_a_supp = DataManager.get_historical_prices(start_date, end_date, tickers, curr)
             
         return other_US_data       
                
                             
         

    
          
