import numpy as np

class Tank():
    def __init__(self, tank_id, current_load, max_load, consumption_rate, n_discrete_load_levels,
                load_level_percentages):
        self.id = tank_id
        self.load = current_load
        self.max_load = max_load
        self.levels = np.linspace(0,self.max_load, n_discrete_load_levels+1)[1:]
        self.level_percentages = load_level_percentages
        
        self.rate =  self.max_load * (self.level_percentages[0]+self.level_percentages[1])/2.0 #consumption rate
   
    def fill(self):
        self.load = self.max_load    
        
    def partial_fill(self, fill_percentage):
        self.load = self.load + self.max_load * fill_percentage
    
    def tank_extra_capacity(self):
        return(self.max_load - self.load)
       
    def is_below_last_level(self):
        if self.load <= self.levels[0]:
            return(True)
        else:
            return(False)
        
    def is_empty(self):
        if self.load <= 0:
            return(True)
        else:
            return(False)
    
    def consume(self):
        self.load = max(0, self.load - self.rate)
       
    def load_to_lvl(self):
         """
        Convert the current load of the tank to the corresponding dicretized level
        """
        levels = self.levels
        lvl = np.amin(np.where(np.isin(levels,levels[ (levels >= self.load) ])))
        if lvl < 0:
            raise ValueError('tank level is negative')
        return(lvl)
      
    def lvl_to_load(self, lvl):
        # Warning: this could be a load different to the one the truck has, since when discretizing in levels
        # we lose information depending on how width the partition intervals are.
        if lvl < 0:
            raise ValueError('tank level is negative')
        else:
            return(self.levels[lvl])
        
        