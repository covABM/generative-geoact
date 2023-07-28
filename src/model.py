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




def get_agent_description_prompt(agent):
    prompt = "Your name is {}, you are a {}-years-old {} from {}, CA."
    prompt_text = "Tell me something more about yourself. Use at most 30 words to explain."
    prompt = prompt.format(agent.unique_id, 
                           agent.background['age'], 
                           agent.background['gender'], 
                           agent.background['home'])
    
    return prompt, prompt_text


def get_agent_plan_prompt(agent):


    prompt = "You are {}. The following is your description: {}"
    promp_text = " You just woke up. What is your goal for today? Write it down in an hourly basis, starting at 9:00. Write only one or two very short sentences. Be very brief. Use at most 50 words."
    prompt = prompt.format(agent.unique_id, agent.description)

    return prompt, prompt_text
    


class GenerativeModel(Model):
    
    schedule_types = {"Sequential": BaseScheduler,
                  "Random": RandomActivation,
                  "Simultaneous": SimultaneousActivation}
    
    
    def __init__(self, agent_class, background_df, agent_N, init_patient=1, schedule_type="Simultaneous"):
        '''
        initialize the model with a GeoSpace grid
        agent_class: the type of agent you want to initialize in this model
                     normally not an input parameter as models are mostly tied to specific agent types
                     here we want to reuse thi model later
        agent_N: number of agents to intialize the model with
        '''
        self.schedule_type = schedule_type
        self.init_patient = init_patient
        
        
        # mesa required attributes
        self.running = True # determines if model should keep on running
        # should be specified to false when given conditions are met
        
        self.grid = GeoSpace() 
        self.schedule = self.schedule_types[schedule_type](self)
        
        self.time_stamps = ["9:00am","10:00am","11:00am","12:00pm","1:00pm","2:00pm","3:00pm","4:00pm"]   
        self.day = 0
        # init agents
        
        ### hard coded room agents
        # should read school map in next udpate
        coords = ((-10, -10), (-10, 10), (10, 10), (10, -10),(-10, -10))
        room_shape = Polygon(coords)
        
        room_agent = Room(model=self,
                          shape=room_shape,
                          unique_id = "r0",
                          room_type="Ball Room")
        self.grid.add_agents(room_agent)
        self.schedule.add(room_agent)
        
        for i in range(agent_N):
            pnt = Point(np.random.uniform(-10, 10), np.random.uniform(-10, 10))
            
            a = agent_class(model=self, 
                            shape=pnt, 
                            agent_background = background_df.loc[i], unique_id="p" + str(i))
            a.room = room_agent
            
            
            # format prompts for language model
            prompt_background, prompt_text = get_agent_description_prompt(a)
            # generate text
            # add text to agent 
            a.description = generate_text(prompt_background, prompt_text)
            self.grid.add_agents(a)
            self.schedule.add(a)
            
        # generate agent description and plan
        


            
        # do the above for plan (plan prompt requires description text, hence this step must be processed after)
        #plan_prompt = [get_agent_plan_prompt(a) for a in self.schedule.agents]
        #plans = language_model(plan_prompt, do_sample=True, min_length=10, max_length=128)
        #for i, agent in enumerate(self.schedule.agents):
        #    agent.plan = plans[i]["generated_text"]
   

    def add_N_patient(self, N): 
        patients = random.sample([a for a in self.schedule.agents if isinstance(a, Human)], N)
        for p in patients:
            p.health_status = "exposed"
            p.infective = True
        
        
    def step(self, keep_memory=True):
        '''
        step function of the model that would essentially call the step function of all agents
        '''
        # openai RPM is 3 currently
        # running more than 3 steps per minute will run into error
        #if self.schedule.steps != 0:
        #    if not (self.schedule.steps%3):
        #        print("Waiting for OpenAI! Sleep 1 min.")
        #        time.sleep(60)
        
        self.keep_memory = keep_memory
        if not self.schedule.steps:
            self.add_N_patient(self.init_patient)
            
            
        if self.schedule.steps%len(self.time_stamps):
            self.day += 1
            print(self.day)
                
        self.time = self.time_stamps[self.schedule.steps%len(self.time_stamps)]
        self.schedule.step()
        self.grid._recreate_rtree() # this is some history remaining issue with the mesa-geo package
        # what this does is basically update the new spatial location of the agents to the scheduler deliberately