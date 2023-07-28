import sys
import os
sys.path.append('./src')
os.chdir(os.path.dirname(sys.path[0]))

# mesa imports
from mesa_geo import GeoAgent, GeoSpace
from mesa.time import BaseScheduler, RandomActivation, SimultaneousActivation
from mesa import datacollection
from mesa import Model

# language model imports
import re
import torch
from transformers import pipeline
import networkx as nx
import openai

# shapely imports
from shapely.geometry import Polygon, Point, LineString
import shapely

# data analysis imports
import geopandas as gpd
import pandas as pd
import numpy as np
import random
import copy
import time


# Configuration Data and Files
import configparser
from scipy import stats

import transmission_rate as trans_rate



# Prefix for config data
#os.chdir(os.path.dirname(sys.path[0]))
config_file_path_prefix = './config/'


# parser viz config data
viz_ini_file = 'vizparams.ini'

parser_viz = configparser.ConfigParser()
parser_viz.read(config_file_path_prefix + viz_ini_file)

default_section = parser_viz['DEFAULT_PARAMS']


# parser disease config data

disease_params_ini = 'diseaseparams.ini'
parser_dis = configparser.ConfigParser()
parser_dis.read(config_file_path_prefix + disease_params_ini)
incubation = parser_dis['INCUBATION']


# NPI config data


npi_params_ini = 'NPI.ini'
parser_npi = configparser.ConfigParser()
parser_npi.read(config_file_path_prefix + npi_params_ini)



# school config data
school_params_ini = 'schoolparams.ini'
parser_school = configparser.ConfigParser()
parser_school.read(config_file_path_prefix + school_params_ini)
population_config = parser_school['SCHOOL_POPULATION']
school_intervention_params = parser_school['INTERVENTION']






# infectious curve config
###################################### 
# based on gamma fit of 10000 R code points

shape, loc, scale = (float(incubation['shape']), float(incubation['loc']), float(incubation['scale']))

# infectious curve
range_data = list(range(int(incubation['lower_bound']), int(incubation['upper_bound']) + 1))
infective_df = pd.DataFrame(
    {'x': range_data,
     'gamma': list(stats.gamma.pdf(range_data, a=shape, loc=loc, scale=scale))
    }
)
#########################################



class Human(GeoAgent):
    '''
    A simple geo-agent that each step moves in the range of [-5,5) and greets all agents within 2 units
    unique_id: the unique_id of the agent
    model: the model that the agent belongs to
    shape: the spatial shape of the agent
    '''
    
    
    
    MEMORY_LIMIT=3
    
    
    
    def __init__(self, unique_id, model, shape, agent_background, crs=3857):
        super().__init__(unique_id, model, shape, crs)
        self.greeted = False # some attribute to indicate if one agent has ever greeted another agent in its life-span
        
        # coordinate attributes
        # eventhough it is not ncessary to establish as attributes
        # setting as attribute would enhance data collection efficiency
        # check mesa documentation for more details in the DataCollector page
        self.x = self.geometry.x
        self.y = self.geometry.y
        
        # set up interaction background for language model
        self.background = agent_background
        self.room = None
        
        
        # human agent memory
        self.current_action = None
        self.memory_ratings = []
        self.memories = []
        self.compressed_memories = []
        
        
        
        
        # mask setup
        # defualt to no mask
        self.mask_type = None
        self.mask_passage_prob = trans_rate.return_mask_passage_prob(self.mask_type)
        
        # disease config

        self.health_status = 'healthy'
        prevalence = float(parser_dis['ASYMPTOMATIC_PREVALENCE']['prevalence'])
        self.asymptomatic = np.random.choice([True, False], p = [prevalence, 1-prevalence])
        self.symptoms = False


        # TODO: vaccination should be parameterized (effective rate, etc.)
        self.tested = False
        self.vaccinated = False
        
        self.infective = False
        
        # symptom onset countdown config
        ##########################################
        # From 10000 lognorm values in R
        countdown = parser_dis['COUNTDOWN']
        shape, loc, scale =  (float(countdown['shape']), float(countdown['loc']), float(countdown['scale']))

        lognormal_dist = stats.lognorm.rvs(shape, loc, scale, size=1)

        
        num_days = max(int(countdown['lower_bound']), 
                       min(np.round(lognormal_dist, 0)[0], 
                           int(countdown['upper_bound'])))# failsafe to avoid index overflow
        self.symptom_countdown = int(num_days)
        #######################################
        
        # default breathing attributes for transmission models
        self.breathing_rate = 'resting'
        self.breathing_activity = 'open_windows'
    
        
    def move(self, other_agent=None, move_range=1):
        '''
        update the current location 
        if other_agent is provided agent will move toward the other agent
        else, the agent will move within the random range and greet to all surrounding neighbors with specified greet_dist
        '''
        

        if other_agent:    
            self.update_shape(other_agent.geometry)
        
        
        move_spread = self.room.geometry.intersection(self.geometry.buffer(move_range))
        

        minx, miny, maxx, maxy = move_spread.bounds 

        while True:
            pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))            
            # check if point lies in true area of polygon
            if move_spread.contains(pnt):
                self.update_shape(pnt)
                break
                
                
    def update_memories(self):
        other_agents = [a for a in self.room.occupants if a.unique_id != self.unique_id]
        for agent in other_agents:
            self.memories.append('[Time: {}. Person: {}. Memory: {}]\n'.format(str(self.model.time), 
                                                                               agent.unique_id, 
                                                                               agent.current_action))

                

    def rate_memories(self):


        memory_ratings = []
        for memory in self.memories:
            prompt = "You are {}. You are currently in {}. It is currently {}:00. You observe the following: {}. Give a rating, between 1 and 5, to how much you care about this."
            prompt = prompt.format(self.unique_id, self.room.room_type, self.model.time, memory)
            res = generate_text("", prompt)
            rating = get_rating(res)
            max_attempts = 2
            current_attempt = 0
            while rating is None and current_attempt < max_attempts:
                rating = get_rating(res)
                current_attempt += 1
            if rating is None:
                rating = 0
            memory_ratings.append((memory, rating, res))
        self.memory_ratings = memory_ratings
        
        
    
    
    def compress_memories(self):


        memories_sorted = sorted(self.memory_ratings, key=lambda x: x[1], reverse=True)
        relevant_memories = memories_sorted[:self.MEMORY_LIMIT]
        memory_string_to_compress = '.'.join([a[0] for a in relevant_memories])
        self.compressed_memories.append('[Recollection at Day {}, Time {}:00: {}]'.format(
            self.model.day, 
            self.model.time, 
            memory_string_to_compress))
        
        
        
        
    @staticmethod
    def droplet_infect(infected, uninfected):
        '''
        baseline transmission rate
        '''
        feet_to_meter = 1/3.2808
        distance = infected.geometry.distance(uninfected.geometry)*feet_to_meter
        
        # normalize symptom countdown value to infectious distribution value
        # 0 being most infectious
        # either -10 or 8 is proven to be too small of a chance to infect others, thus covering asympotmatic case
        
        countdown_norm = min(int(incubation['upper_bound']), max(int(incubation['lower_bound']), 0 - infected.symptom_countdown))
        transmission_baseline = infective_df[infective_df['x'] == countdown_norm]['gamma'].iloc[0]

     
        # Use Chu distance calculation ## see docs
        chu_distance_multiplier = 1/2.02
        distance_multiplier = (chu_distance_multiplier)**distance                                                        

        
        # approximate student time spent breathing vs talking vs loudly talking
        # upperbound baseline (worst case) for breathing activity is moderate_excercise and talking loud
        base_bfr = trans_rate.return_breathing_flow_rate('moderate_exercise')
        base_eai = trans_rate.return_exhaled_air_inf('talking_loud')
        
        
        inf_bfr_mult = trans_rate.return_breathing_flow_rate(infected.breathing_rate)/base_bfr 
        inf_eai_mult = trans_rate.return_exhaled_air_inf(infected.breathing_activity)/base_eai
        
        uninf_bfr_mult = trans_rate.return_breathing_flow_rate(uninfected.breathing_rate)/base_bfr 
        
        # take average of breathing flow rate of two agents
        bfr_multiplier = np.mean([inf_bfr_mult, uninf_bfr_mult])
        # we dont think the uninfected air exahale rate should be a factor here 
        breathing_type_multiplier = bfr_multiplier*inf_eai_mult
        
        

        # Mask Passage: 1 = no masks, .1 = cloth, .05 = N95
        mask_multiplier = np.mean([infected.mask_passage_prob, uninfected.mask_passage_prob])

            


        return transmission_baseline * distance_multiplier * breathing_type_multiplier * mask_multiplier 

                        
    def __check_same_room(self, other_agent):
        '''
        check if current agent and other agent is in the same room
        
        the purpose of this function is to make sure to eliminate edge cases that one agent near the wall of its room
        infects another agent in the neighboring room
        
        this is at this iteration of code only implemented for class purpose, as unique id check is way more efficient
        
        later implementation should add attribute to human agent for current room
        
            other_agent: other agent to check
            returns: boolean value for if the two agents are in the same room
        '''
        return (self.room.unique_id == other_agent.room.unique_id)

    


    
    def update_shape(self, new_shape):
        self.geometry = new_shape
        self.x = self.geometry.x
        self.y = self.geometry.y
    
    def step(self):
        
        # memeory tracking is very time consuming
        if self.model.keep_memory:
            self.update_memories()
            self.rate_memories()
            self.compress_memories()
        
        if self.model.schedule_type != "Simultaneous":
            self.advance()
    
    def advance(self):
        # greet near by agents 
        
        max_infect_dist = 30
        
        neighbors = self.model.grid.get_neighbors_within_distance(self, max_infect_dist)
        # just check in loop
        #neighbors = [neighbor for neighbor in neighbors if neighbor.unique_id != self.unique_id]
        
        
        if self.health_status == 'exposed' and self.infective:
            for neighbor in neighbors:
                if issubclass(type(neighbor), Human) and self.__check_same_room(neighbor) :
                    if neighbor.unique_id != self.unique_id and (neighbor.health_status == 'healthy'):                   
                        # Call Droplet transmission function
                        temp_prob = self.droplet_infect(self, neighbor)
                        infective_prob = np.random.choice ([True, False], p = [temp_prob, 1-temp_prob])
                        if infective_prob and not neighbor.vaccinated:
                            neighbor.health_status = 'exposed'

          
    
    


    
    
class Room(GeoAgent):
    
    
    # dummy config for data collection
    health_status = None
    symptoms = None
    x = None
    y = None
    
    
    def __init__(self, unique_id, model, shape, room_type, crs=3857):
        super().__init__(unique_id, model, shape, crs)
        #self.occupants = []
        #self.aerosol_transmission_rate = []
        #self.barrier = barrier_type
        self.room_type = room_type
        #self.seating_pattern = None
        #self.viral_load = 0
        #self.schedule_id = None
        self.activity = "social gathering event"
        self.occupants = []
        # volume
        #self.floor_area = shape.area
        #self.height = 12

        # airflow ventiliation type
        self.environment = eval(school_intervention_params['ventilation_type'])
    
    def step(self):
        #reset location of agents within the room:
        self.occupants = [a for a in list(self.model.grid.get_intersecting_agents(self)) if issubclass(type(a), Human)]
        
        for a in self.occupants:
            a.move(move_range=10)
        
            
        
        occupant_ids = [a.unique_id for a in self.occupants]
        #exposed = [a for a in occupants if a.infective]

        num_occupants = len(self.occupants)
        #num_exposed = len(exposed)

        # assume this is a cafe for now

        prompt = "There are {} people in this location.".format(num_occupants)

        prompt += ' Currently it is {}.'.format(self.model.time)

        agent_descriptions = [f"{agent.unique_id}: {agent.description}" for agent in self.occupants]
        prompt += ' We know the following about people: ' + '. '.join(agent_descriptions)

        #agent_plans = [f"{agent.unique_id}: {agent.plan}" for agent in self.schedule.agents]
        #prompt += ' We know the daily plan about people: '+ '. '.join(agent_plans)

        prompt += 'They can interact with each other. '

        prompt += "What will each person do in the next hour? Use at most 10 words for each person. Choose one action for each person"
        
        action_text = generate_text("This is a {}. Currently there is a {} happening in this room".format(self.room_type, self.activity), prompt, use_openai=True)
        
        print(action_text)
        
        
        action_lst = re.split(r'p[0-9]+: ', action_text)[1:]
        action_ids = copy.deepcopy(occupant_ids)
        
        for i in range(len(action_lst)):
            act = action_lst[i]
            
            # update agent current action
            self.occupants[i].current_action = act
            # let language model to decide who is having social interaction
            
            # maybe formulate based on level of interaction
            print(act)
            action_bool = generate_text("", "Is there social interaction happening in the action: {} Answer only Yes or No.".format(act))

            if 'No' in action_bool:
                if occupant_ids[i] in action_ids:
                    action_ids.remove(occupant_ids[i])
                
                print('No social interaction for this person!')
                continue
            # if no available actable agents, end loop
            if len(action_ids) == 0:
                break
            actor_id = occupant_ids[i]
            prompt = "The action of {} is: {}. Out of {}, who are interacting with {}. Choose one person.".format(actor_id, 
                                                                                 act, 
                                                                                 ' '.join(action_ids), 
                                                                                 actor_id)

            # let language model decide the actors in this action
            actors_text = generate_text("", prompt)
            
            print("THE ACTORS ARE: ")
            agent_ids = re.findall(r'p[0-9]+', actors_text)
            for agent_id in agent_ids:
                if agent_id in action_ids:
                    action_ids.remove(agent_id)
              
    
            # if more than one agent in action text
            # bring them together physically
            print(agent_ids)
            if len(agent_ids) > 1:
                # get agent objects from agent_ids
                agents = [a for a in self.occupants if a.unique_id in agent_ids] 
                
                # move all agents to location of first agent in list
                for a in agents[1:]:
                    a.move(agents[0])
            
        if self.model.schedule_type != "Simultaneous":
            self.advance()
        
        
        
    def advance(self):
        """
            aerosal transmission model
        """
        self.model.grid._recreate_rtree() 
 
        exposed = [a for a in self.occupants if a.infective]

        num_occupants = len(self.occupants)
        num_exposed = len(exposed)


        exposure_time = 1

        mean_breathing_rate = np.mean([trans_rate.return_breathing_flow_rate(a.breathing_rate) for a in self.occupants])
        mean_infectivity = np.mean([trans_rate.return_exhaled_air_inf(a.breathing_activity) for a in self.occupants])
        ACH = trans_rate.return_air_exchange_rate(self.environment)
        floor_area = self.floor_area
        
        
        # TODO: take mean of mask_prob of all human agents in the room
        mask_passage_prob = np.mean([a.mask_passage_prob for a in occupants])
        height = self.height


        transmission_rate = aerosol_new.return_aerosol_transmission_rate(floor_area=floor_area, room_height=height,
        air_exchange_rate=ACH, aerosol_filtration_eff=0, relative_humidity=0.69, breathing_flow_rate=mean_breathing_rate, exhaled_air_inf=mean_infectivity,
        mask_passage_prob=mask_passage_prob)

        transmission_rate *= exposure_time
        transmission_rate *= num_exposed #To be changed to some proportion to get infectious
        


        self.aerosol_transmission_rate.append(transmission_rate)
        
            
        
    def generate_seats(self, N, width):
        
        self.seating_pattern = style
        self.seats = []
        shape = self.geometry
        
        
        # generate grid seating that seperates each student by fixed amount

        center = shape.centroid
        md = math.ceil(N**(1/2))
        pnt = Point(center.x - width*md//2, center.y - width*md//2)
        for i in range(md):
            for j in range(md+1):
                self.seats.append(Point(pnt.x + i*width, pnt.y + j*width))


    



