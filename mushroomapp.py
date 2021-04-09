# -*- coding: utf-8 -*-
"""
A mushroom classification app, using the Kivy library
and machine learning models.

Created on Wed Mar 31 16:01:10 2021

@author: jarogi
@version: 2.1
"""

# import statements
import kivy
kivy.require('2.0.0')
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import StringProperty

# Start Screen
class StartWindow(Screen):
    pass

# Cap Colour Screen
class CapColorWindow(Screen):
    def ccselection(self,capcol):
        #capcolout = "You clicked the "+capcol+" button!"
        self.manager.my_cap_color = capcol
        #if capcol == 'blue':
            #self.manager.my_cap_color = 'BLUE'
        #self.ids.result0.text=capcolout

# Bruises Window
class BruisesWindow(Screen):
    def bselection(self, bruises):
        self.manager.my_bruises = bruises
        
# Gill Attachment Window
class GillAttachmentWindow(Screen):
    def gaselection(self,gillatt):
        self.manager.my_gill_attachment = gillatt
        
# Stalk Root Window
class StalkRootWindow(Screen):
    def srselection(self, sroot):
        self.manager.my_stalk_root = sroot
        
# Habitat Window
class HabitatWindow(Screen):
    def hselection(self, habitat):
        self.manager.my_habitat = habitat

# Summary Window
class SummaryWindow(Screen):
    pass

# Window Manager
class WindowManager(ScreenManager):
    my_cap_color = StringProperty('')
    my_cap_shape = StringProperty('')
    my_bruises = StringProperty('')
    my_gill_attachment = StringProperty('')
    my_stalk_root = StringProperty('')
    my_habitat = StringProperty('')

# load in kivy file, to stylize GUI
kv = Builder.load_file("mushroom.kv")

class MushroomApp(App):
    def build(self):
        return kv
    
if __name__ == '__main__':
    MushroomApp().run()
