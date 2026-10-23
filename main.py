
import time as time
from DataHandler import DataHandler
from Coach import Coach
from Log import setup_logger
# import LightGCN
if __name__ == "__main__":
    # LightGCN.main()
    setup_logger()
    handler = DataHandler()
    coach = Coach(handler)
    coach.run()