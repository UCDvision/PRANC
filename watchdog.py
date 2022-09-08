import time

class WatchDog():
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
    
    def get_time_in_sec(self):
        return round(self.end_time - self.start_time, 2)
    
    def get_time_in_ms(self):
        return round ((self.end_time - self.start_time) * 1000, 2)