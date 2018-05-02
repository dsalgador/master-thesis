import numpy as np

class Truck():
    def __init__(self, truck_id, current_load, max_load, current_position, load_fractions_deliverable,
                n_discrete_load_levels):
        self.id = truck_id
        self.load = current_load
        self.max_load = max_load
        self.pos = current_position
        self.fractions = load_fractions_deliverable
        self.levels = np.linspace(0,self.max_load, n_discrete_load_levels+1)[1:]
     
    def fill(self):
        self.load = self.max_load
        
    def deliver(self, fraction_id: int):
        self.load = self.load - self.fractions[fraction_id] * self.max_load
    
    def possible_delivery_quantities(self, tank_extra_capacity):
        all_delivery_quantities = self.load * self.fractions
        return(all_delivery_quantities[ all_delivery_quantities <= tank_extra_capacity].astype(list))
    
#     def load_to_lvl(self):
#         levels = self.levels
#         #print("truck", np.where(np.isin(levels,levels[ (levels >= self.load) ])))
#         lvl = np.amin(np.where(np.isin(levels,levels[ (levels >= self.load) ])))-1
#         if(lvl == -1):
#             return(0)
#         else:
#             return(lvl)
    
#     def lvl_to_load(self, lvl):
#         # Warning: this could be a load different to the one the truck has, since when discretizing in levels
#         # we lose information depending on how width the partition intervals are.
#         if lvl == -1:
#             return(0)
#         else:
#             return(self.levels[lvl])
        
    def load_to_lvl(self):
        levels = self.levels
        #print(np.where(np.isin(levels,levels[ (levels >= self.load) ])))
        #print("levels", levels)
        #print("self.load, self.max_load", self.load, self.max_load)
        lvl = np.amin(np.where(np.isin(levels,levels[ (levels >= self.load) ])))
        if lvl < 0:
            raise ValueError('tank level is negative')
        #if(lvl == -1):
            #return(0)
        #else:
        return(lvl)
      
    def lvl_to_load(self, lvl):
        # Warning: this could be a load different to the one the truck has, since when discretizing in levels
        # we lose information depending on how width the partition intervals are.
        if lvl < 0:
            raise ValueError('tank level is negative')
        else:
            return(self.levels[lvl])
        
          
    