# -*- coding: utf-8 -*-
"""
A mushroom classification app, using the Kivy library
and machine learning models.

Created on Wed Mar 31 16:01:10 2021

@author: jarogi
@version: 3.0
"""

# import statements
import kivy
kivy.require('2.0.0')
import numpy as np
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import StringProperty
from joblib import load

# Start Screen
class StartWindow(Screen):
    pass

# Cap Colour Screen
class CapColorWindow(Screen):
    def ccselection(self, capcol, cc):
        self.manager.my_cap_color = capcol
        self.manager.cap_color = cc

# Bruises Window
class BruisesWindow(Screen):
    def bselection(self, bruises, b):
        self.manager.my_bruises = bruises
        self.manager.bruises = b
        
# Gill Attachment Window
class GillAttachmentWindow(Screen):
    def gaselection(self, gillatt, ga):
        self.manager.my_gill_attachment = gillatt
        self.manager.gill_attachment = ga
        
# Stalk Root Window
class StalkRootWindow(Screen):
    def srselection(self, sroot, sr):
        self.manager.my_stalk_root = sroot
        self.manager.stalk_root = sr
        
# Habitat Window
class HabitatWindow(Screen):
    def hselection(self, habitat, h):
        self.manager.my_habitat = habitat
        self.manager.habitat = h

# Summary Window
class SummaryWindow(Screen):
    def predictmush(self):
        # create an array to for model input
        mush_features = np.array([self.manager.cap_color, 
                                  self.manager.bruises, 
                                  self.manager.gill_attachment, 
                                  self.manager.stalk_root, 
                                  self.manager.habitat])
        print(mush_features)
        
        # check if any features are missing
        missing_feature = False
        predict_out = ''
        for feat in mush_features:
            if feat == '':
                print("Missing Feature")
                predict_out = "Missing Feature(s), please go back and select an option for all features."
                missing_feature = True
                break
        
        # run model prediction if all features assigned
        if not missing_feature:
            # encode input
            encoded_input = self.manager.my_encoder.transform(mush_features.reshape(1, -1))
            # have model classify input
            is_poisonous = self.manager.my_model.predict(encoded_input)[0]
            if is_poisonous:
                predict_out = "Model Prediction: Poisonous"
            else:
                predict_out = "Model Prediction: Edible"
        self.ids.prediction.text = predict_out
    def clearpredict(self):
        self.ids.prediction.text = ''

# Window Manager
class WindowManager(ScreenManager):
    # initalize mushroom feature variables (full name)
    my_cap_color = StringProperty('')
    my_bruises = StringProperty('')
    my_gill_attachment = StringProperty('')
    my_stalk_root = StringProperty('')
    my_habitat = StringProperty('')
    
    # initialize msuhroom feature (single character)
    cap_color = StringProperty('')
    bruises = StringProperty('')
    gill_attachment = StringProperty('')
    stalk_root = StringProperty('')
    habitat = StringProperty('')
    
    # load in machine learning model and encoder
    my_model = load("mushmodel.joblib")
    my_encoder = load("encoder.joblib")

# load in kivy file, to stylize GUI
kv = Builder.load_file("mushroom.kv")

class MushroomApp(App):
    def build(self):
        return kv
    
if __name__ == '__main__':
    MushroomApp().run()
