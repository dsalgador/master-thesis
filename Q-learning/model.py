import numpy as np
import math
import networkx as nx
import random

import matplotlib.pyplot as plt
from tank import Tank
from truck import Truck
import copy


import gym_pdsystem.utils.utilsq as ut
import gym_pdsystem.utils.constants as ct
import gym_pdsystem.utils.functions as fnc


COEFF = ct.COEFF

C_TRANSPORT = ct.C_TRANSPORT
C_LEVELS = ct.C_LEVELS

p0_GLOBAL = ct.p0_GLOBAL

P1_GLOBAL = ct.P1_GLOBAL
P2_GLOBAL = ct.P2_GLOBAL

M_GLOBAL = ct.M_GLOBAL

NOT_DELIVERYING_PENALTY = P2_GLOBAL #to be equivalent/same importance as having 0 stock or surpassing max capacity levels

    
class System():
    def __init__(self, tanks, trucks, adjacency_matrix, weights_matrix):
        self.tanks = tanks
        self.trucks = trucks
        self.graph = adjacency_matrix
        self.weights = weights_matrix
        self.k = len(trucks)
        self.n = len(tanks)
        self.s = self.state()
        self.ds = self.discrete_state()
        
        self.tanks_id = self.tank_ids()
        self.trucks_id = self.truck_ids()
        self.tanks_max_load = self.tank_max_loads()
        self.trucks_max_load = self.truck_max_loads()
        self.tanks_level = self.tank_levels()
        self.trucks_level = self.truck_levels()
        
        #
        self.actions_dim = self.n * self.k
        self.states_dim = self.n_states()
        self.state_length = 2*self.k + self.n
        self.action_length = 2*self.k
        self.state_action_length = self.state_length + self.action_length

        #
        self.a = None
        self.da = None
        
    def get_trucks(self):
            return(self.trucks)
    
    def n_states(self):
        """
        Return the total number of possible states that the system of trucks and tanks can have.
        """
        n_s = 1
        possible_fractions_deliverable = []
        for i,truck in enumerate(self.trucks):
               possible_fractions_deliverable.append(len(truck.fractions))
        
        n_s = n_s * (self.n+1)**(self.k) 
        
        for i in range(self.k):
            n_s = n_s * (possible_fractions_deliverable[i])
        
        n_s = n_s * np.prod([len(self.tanks_level[i]) for i in range(self.n)])
                       
        return(n_s)    
               
        
    def truck_loads(self):
        return([self.trucks[i].load for i in range(self.k)])
    
    def truck_max_loads(self):
        return([self.trucks[i].max_load for i in range(self.k)])
    
    def truck_positions(self):
        return([self.trucks[i].pos for i in range(self.k)])
    
    def truck_ids(self):
        return([self.trucks[i].id for i in range(self.k)])
    
    def truck_levels(self):
        return([self.trucks[i].levels for i in range(self.k)])
    
    
    
    
    def tank_loads(self):
        return([self.tanks[i].load for i in range(self.n)])
    
    def tank_max_loads(self):
        return([self.tanks[i].max_load for i in range(self.n)])
    
    def tank_ids(self):
        return([self.tanks[i].id for i in range(self.n)])
    
    def tank_levels(self):
        return([self.tanks[i].levels for i in range(self.n)])

        
    def state(self):
        """
        Returns the current (continuous) state of the system
        """
        s = [self.truck_positions(), self.truck_loads(), self.tank_loads()]
        return(s)
    
    def discrete_state(self):
        """
        Returns the current discretized state version of the system
        """
        ds = [ [],[] ,[] ]
        ds[0] = self.s[0]
        
        for i, truck in enumerate(self.trucks):
            ds[1].append(truck.load_to_lvl())
            
            
        for i, tank in enumerate(self.tanks):
            ds[2].append(tank.load_to_lvl())
            
        return(ds)    
            
    def set_discrete_state(self, d_state):
        """
        Update the current discretized state version of the system
        """

        ds = [ [],[] ,[] ]

        for i in d_state[:self.k]:
            ds[0].append(int(i))
        for i in d_state[self.k:(2*self.k)]:
            ds[1].append(int(i))
        for i in d_state[2*self.k:]:
            ds[2].append(int(i))
            
        self.ds = ds  
        
        
    def update_state(self):
        """
        Update both the discretized and contiuous state of the system (just in case some tank or truck levels,
        truck positions, etc. have been modified/updated)
        """
        self.s = self.state()
        self.ds = self.discrete_state()
        
    def state_to_string(self):
        """
        Returns a string that encodes the current discrete state of the system
        """
        state_str = ''.join(str(''.join(str(y) for y in x)) for x in self.ds)
        return(state_str)
    
    def action_to_string(self):
        """
        Returns a string that encodes the current discrete action taken by the system
        """
        action_str = ''.join(str(''.join(str(y) for y in x)) for x in self.da)
        return(action_str)
        
    def state_action_to_string(self):
        """
        Returns a string that encodes the current discrete state-action of the system
        """
        state_str = ''.join(str(''.join(str(y) for y in x)) for x in self.ds)
        action_str = ''.join(str(''.join(str(y) for y in x)) for x in self.da)
        sa_str = ''.join([state_str, action_str])
        if len(sa_str) != self.state_action_length:
            print("da", self.da)
            print("ds", self.ds)
            print("sa_str", sa_str)
            raise ValueError('sa_str of wrong length in state_action_to_string()')
            
        return(sa_str)
            
    def visualize(self, show = False):
            """
            TO DO
            """
            index = np.arange(self.n)
            tanks_max_load = self.tanks_max_load
            tank_loads = self.tank_loads()
            tanks_id = self.tanks_id
            
            plt.bar(index, tanks_max_load, color = 'black')
            plt.bar(index, tank_loads, color = 'blue' )
            plt.xlabel('Tank id', fontsize=10)
            plt.ylabel('Current level', fontsize=10)
            plt.xticks(index, tanks_id, fontsize=10, rotation=30)
            plt.title('Current tanks state')
            
            if show:  plt.show()
            
            return([index, tanks_max_load, tank_loads, tanks_id])
        
    def visualize_step(self, args): 
            """
            TO DO
            """
            index, tanks_max_load, tank_loads, tanks_id = args;
            plt.bar(index, tanks_max_load, color = 'black')
            plt.bar(index, tank_loads, color = 'blue' )
            plt.xlabel('Tank id', fontsize=10)
            plt.ylabel('Current level', fontsize=10)
            plt.xticks(index, tanks_id, fontsize=10, rotation=30)
            plt.title('Current tanks state')
        
    
    def is_some_tank_empty(self):
        """
        Returns a boolean saying if there is some tank empty (i.e., with load less than zero)
        """
        for tank in self.tanks:
            if tank.is_empty():
                return(True)
        return(False)    
    
    def number_of_tanks_empty(self):
        """
        Returns the number of tanks that are empty (i.e., with load less than zero)
        """
        i = 0
        for tank in self.tanks:
            if tank.is_empty():
                i = i+1
        return(i)    
    
    def is_some_tank_below_last_level(self):
        """
        Returns True if there is some tank whose current load correspond to the lower (last) discrete load level.
        """
        for tank in self.tanks:
            if tank.is_under_last_level():
                return(True)
        return(False)    
    
    def number_of_tanks_below_last_level(self):
        """
        Returns the number of tanks whose current load correspond to the lower (last) discrete load level.
        """

        i = 0
        for tank in self.tanks:
            if tank.is_below_last_level():
                i = i+1
        return(i)    
  
    
    def reset_trucks_positions(self):
        for truck in self.trucks:
            truck.pos = self.n
            
    def reset_trucks_loads(self):
        for truck in self.trucks:
            truck.load = truck.max_load  
            
    def R_transport(self, coeff, w, u):
        return( coeff * np.sum(w*u) )
    
    def R_levels(self, p0 = p0_GLOBAL, M = M_GLOBAL, P1 = P1_GLOBAL,  P2 = P2_GLOBAL): #STILL TO DECIDE THE DEFAULT VALUES 
        
        R = 0
        
        for i, tank in enumerate(self.tanks):
            percentages = tank.level_percentages
            b = percentages[0]
            c = percentages[1]
            e = percentages[2]
            
            a = b/10
            f = 1-(1-e)/10
            d = p0*e+(1-p0)*c
            
            C_max = tank.max_load
            x = tank.load
            
            R = R + fnc.R_lvl(x, C_max, a,b,c,d,e,f,P1,P2,M)
            
            
        return(R)  
            
    
    def random_action(self, seed = None, verbose = False):
        """
        TO DO
        """
        #It is assumed that the current state of the system is updated.
        
        if seed != None:
            random.seed(seed)
            
            
        new_positions = [] 
        new_deliveries = []
        new_deliveries_index = []
        
        w_t = np.full(self.k, 0)
        trucks_not_deliverying = 0
        
        rewards = 0  
        #print("self.trucks",self.get_trucks())
            
        # Choose a position for each truck randomly
        
        for i, truck in enumerate(self.get_trucks()):
            old_position = truck.pos
            if verbose: print("truck pos: ", old_position)
            possible_positions_index = np.isin(self.graph[old_position], 1)
            possible_positions = np.where(possible_positions_index)
            #random.randint(0,len(possible_positions[0])-1)
            if verbose: print("nº of possible positions", len(possible_positions_index))
            new_position = random.randrange(len(possible_positions_index))
            if verbose: print("new position: ",new_position)
            truck.pos = new_position
            if verbose: print("possible_positions:", possible_positions)
            
            # Update rewards due to oil costs (transport/km)
            w_t[i] = self.weights[old_position][new_position]
            new_positions.append(new_position)

            
            
        # Choose a new (possible) load delivery for each truck to the new tank (position)
        # and update the tank's load after deliverying the chosen quantity.
        
        for i, current_truck in enumerate(self.get_trucks()):
                truck_pos = current_truck.pos
                
                if truck_pos != self.n:
                    if verbose: print("truck_pos: ", truck_pos)

                    #current_truck = truck
                    current_tank = self.tanks[truck_pos]
                    current_extra_tank_capacity = current_tank.tank_extra_capacity()
                    possible_delivery_quantities = current_truck.possible_delivery_quantities(current_extra_tank_capacity)
                    if verbose: print("Possible delivery quantities: ", possible_delivery_quantities)
                    if possible_delivery_quantities.size == 0:
                        if verbose: print(f"Truck {truck.id} in tank {truck.pos} does not deliver")
                        random_index = len(current_truck.levels)-1 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! problem dependent
                        #alomejor habria que poner len(current_truck.levels)+1 , ya que si no esto es como poner 0 (en el caso
                        #de solo un nivel de carga de camión) y el () + 1 añadiria la acción "no descargar nada"
                        delivery_quantity = 0
                        #rewards = rewards -10**3 #-np.inf
                        trucks_not_deliverying = trucks_not_deliverying + 1 
                        
                    else:
                        random_index = random.randint(0,len(possible_delivery_quantities)-1)
                        #delivery_quantity = np.random.choice(possible_delivery_quantities)
                        delivery_quantity = possible_delivery_quantities[random_index]
                        current_tank.load = current_tank.load + delivery_quantity
                        if verbose: print(f"Truck {truck.id} in tank {truck.pos} delivers {delivery_quantity} units")

                        #rewards = rewards - delivery_quantity
                        
                else:
                    delivery_quantity = 0
                    random_index = 0 
                                        
                new_deliveries.append(delivery_quantity)
                new_deliveries_index.append(random_index)
    
        u_t = np.asarray(new_deliveries)             
        
        # Update the loads of the tanks accordig to their consumption rates
        for tank in self.tanks:           
            tank.consume()
        
        
        self.da = [new_positions, new_deliveries_index]
        self.a = [new_positions, new_deliveries]
        
        if len(self.action_to_string()) != self.action_length:
            print("ACTION WITH WRONG LENGTH")
            
        transport_rewards = C_TRANSPORT * self.R_transport(COEFF, w_t, u_t)
        level_rewards = C_LEVELS * self.R_levels()
        
        rewards = level_rewards - transport_rewards + C_LEVELS *trucks_not_deliverying * NOT_DELIVERYING_PENALTY
        
        return(rewards, transport_rewards, level_rewards)
    
    
    def deterministic_action(self, action, verbose = False):
        """
        TO DO
        """
        
        #It is assumed that the current state of the system is updated.

        rewards = 0
        def action_to_int(action):
            int_action = []
            for i in action:
                int_action.append(int(i))
            return(int_action)
        
        new_positions = [] 
        new_deliveries = []
        new_deliveries_index = []
        
        w_t = np.full(self.k, 0)

        
        action = action_to_int(action)
        
        for i, new_position in enumerate(action[0:self.k]):
            old_position = self.trucks[i].pos
            self.trucks[i].pos = new_position
            w_t[i] = self.weights[old_position][new_position]
            new_positions.append(new_position)

     
        for new_delivery_index, truck in zip(action[self.k:], self.trucks):
            if truck.pos != self.n:
                current_tank = self.tanks[truck.pos]
                delivery_quantity = truck.lvl_to_load(new_delivery_index)
                truck.load = truck.load - delivery_quantity

                current_tank.load = min(current_tank.load + delivery_quantity, current_tank.max_load)
                #if the current unload is about to overfill the tank, we only fill it up intil its max capacity
                #note that this would be a bit in contradiction with the assumption that the trucks go full and return empty,
                #but this is needed because of the discretization of the problem.
                #rewards = rewards - delivery_quantity
            else:
                delivery_quantity = 0
                new_delivery_index = 0 
                
            new_deliveries_index.append(new_delivery_index)
            new_deliveries.append(delivery_quantity)

        u_t = np.asarray(new_deliveries)             

        # Update the loads of the tanks accordig to their consumption rates
        for tank in self.tanks:           
            tank.consume()
        
        self.da = [new_positions, new_deliveries_index]
        self.a = [new_positions, new_deliveries]    
       
        if len(self.action_to_string()) != self.action_length:
            print("ACTION WITH WRONG LENGTH")
            
        if verbose: print(self.da, self.a)
            
        transport_rewards = C_TRANSPORT * self.R_transport(COEFF, w_t, u_t)
        level_rewards = C_LEVELS * self.R_levels()
            
        rewards = level_rewards - transport_rewards #C_LEVELS * self.R_levels() - C_TRANSPORT * self.R_transport(COEFF, w_t, u_t) 
                  #+ trucks_not_deliverying * NOT_DELIVERYING_PENALTY
        
        
        return(rewards, transport_rewards, level_rewards)
        
            
            
            
            
            