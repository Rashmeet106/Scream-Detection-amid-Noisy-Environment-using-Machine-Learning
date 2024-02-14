import datetime
import sounddevice as sd
from kivy.uix.checkbox import CheckBox
from kivymd.uix.bottomsheet import MDListBottomSheet
import shutil
from scipy.io.wavfile import write
import pandas as pd
import numpy as np
import pytz
import requests
from kivymd.toast import toast
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.app import App
from kivymd.icon_definitions import md_icons
from kivy.clock import Clock
from kivy.graphics.opengl import *
from kivy.graphics import *
from kivy.properties import ListProperty, ObjectProperty, NumericProperty
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.core.text import LabelBase
from kivymd.uix.button import MDFlatButton
from kivy.uix.textinput import TextInput
import threading
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivymd.uix.dialog import MDDialog
import os, threading, time
from kivymd.app import MDApp
from kivymd.uix.button import MDRectangleFlatButton
from kivy.uix.behaviors import ButtonBehavior

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense


class ImageButton(ButtonBehavior, Image):
    pass


class NavigationLayout(BoxLayout):
    pass


class MDToolbar(BoxLayout):
    pass


class MainWindow(BoxLayout):
    pass


class HelpWindow(BoxLayout):
    pass


class PopupWarning(BoxLayout):
    label_of_emergency = ObjectProperty(None)


def mic_clicked(instance):
    print("hello")
    app = App.get_running_app()
    app.mic_clicked()


class AudioRecWindow(BoxLayout):
    micbutton = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(AudioRecWindow, self).__init__(**kwargs)
        self.micbutton.bind(on_release=mic_clicked)


class ContentNavigationDrawer(BoxLayout):
    pass


class InternalStorageWindow(BoxLayout):
    pass


class TeamWindow(BoxLayout):
    pass


class FileLoader(BoxLayout):
    filechooser = ObjectProperty(None)


class uiApp(MDApp):
    dialog = None

    def build(self):
        self.theme_cls.primary_palette = "Pink"

        self.theme_cls.theme_style = "Dark"  # "Light"

        self.screen_manager = ScreenManager()
        self.mainscreen = MainWindow()
        screen = Screen(name='mainscreen')
        screen.add_widget(self.mainscreen)
        self.screen_manager.add_widget(screen)

        self.recscreen = AudioRecWindow()
        screen = Screen(name='recscreen')
        screen.add_widget(self.recscreen)
        self.screen_manager.add_widget(screen)

        self.internalstoragescreen = InternalStorageWindow()
        screen = Screen(name='internalstoragescreen')
        screen.add_widget(self.internalstoragescreen)
        self.screen_manager.add_widget(screen)

        self.helpscreen = HelpWindow()
        screen = Screen(name='helpscreen')
        screen.add_widget(self.helpscreen)
        self.screen_manager.add_widget(screen)

        self.fileloaderscreen = FileLoader()
        screen = Screen(name='fileloaderscreen')
        screen.add_widget(self.fileloaderscreen)
        self.screen_manager.add_widget(screen)

        self.teamscreen = TeamWindow()
        screen = Screen(name='teamscreen')
        screen.add_widget(self.teamscreen)
        self.screen_manager.add_widget(screen)

        self.popupwarningscreen = PopupWarning()
        screen = Screen(name='popupwarningscreen')
        screen.add_widget(self.popupwarningscreen)
        self.screen_manager.add_widget(screen)

        return self.screen_manager

    def thread_for_rec(self):
        if self.recscreen.micbutton.source == "resources/icons/micon.png":
            fs = 44100  # Sample rate
            seconds = 10  # Duration of recording
            print('rec started')
            self.myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype=np.int16)

            sd.wait()  # Wait until recording is finished
            if self.externally_stopped == True:
                pass
            else:
                write('recorded.wav', fs, self.myrecording)  # Save as WAV file in 16-bit format
                toast("Finished")
                print("stopped")
                self.filename = "recorded.wav"
                # self.recscreen.micbutton.source = "resources/icons/micoff.png"

    def show_popup(self, text):
        show = PopupWarning()
        show.label_of_emergency.text = text
        self.popupWindow = Popup(title="Popup Window", content=show, size_hint=(None, None), size=(400, 400))

        self.popupWindow.open()

    def close_popup(self):
        self.popupWindow.dismiss()

    def process_the_sound(self):
        print("process")
        from modelloader import process_file
        print("p1")
        from svm_based_model.model_loader_and_predict import svm_process
        print("filename")
        print(self.filename)
        output1 = svm_process(self.filename)  # it will process file in svm-model
        
    

        print("model 1")
        print(output1)
        
        df = pd.read_csv("C:\\Users\\harle\\Downloads\\pych_scream\\pych_scream\\scream_detection_other_files_part_2\\newresources.csv", index_col=0, engine = 'c')
        file = open("C:\\Users\\harle\\Downloads\\pych_scream\\pych_scream\\begining index of testing files.txt","r")
        data1 = int(file.read())
        file.close()
        row_num_for_verification_of_model = data1
        X = df.iloc[:row_num_for_verification_of_model,1:]  #independent variables columnns
        print(row_num_for_verification_of_model)
        X2 = df.iloc[row_num_for_verification_of_model:,1:]
        file = open("C:\\Users\\harle\\Downloads\\pych_scream\\pych_scream\\input dimension for model.txt","r")
        data2 = int(file.read())
        file.close()
        print(data2)
        total_number_of_column_required_for_prediction = data2
        column_number_of_csv_having_labels = 0
        y = df.iloc[:data1,column_number_of_csv_having_labels] # dependent variable column
        # # define the keras model
        model = Sequential()
        model.add(Dense(12, input_dim=total_number_of_column_required_for_prediction, activation='relu'))
        model.add(Dense(8, activation='relu'))

        model.add(Dense(10, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(3, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # compile the keras model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # fit the keras model on the dataset
        history = model.fit(X, y,validation_split=0.33, epochs=150, batch_size=50

                            )


        # evaluate the keras model
        _, accuracy = model.evaluate(X, y)
        print('Accuracy: %.2f' % (accuracy * 100))

        # make probability predictions with the model
        predictions = model.predict(X2)

        # round predictions
        rounded = [round(x[0]) for x in predictions]

        print("predicted value is"+str(rounded))
        print("actual value was"+str(list(df.iloc[row_num_for_verification_of_model:,column_number_of_csv_having_labels])))

        model.save('pych_scream')

        output2 = process_file(self.filename)  # it will process file in multilayer perceptron model

        print("model 2")
        print(output2)
        if output1 == True and output2 == True:
            # call emergency funtion with higher risk currently we haven;t implemented emergency function
            text = "[size=30]Risk is [color=#FF0000]high[/color] calling \nemergency function[/size]"
            self.show_popup(text)
            pass
        elif output1 == True or output2 == True:
            # call emergency function
            text = "[size=30]Risk is [color=#008000]Medium[/color] calling \nemergency function[/size]"
            self.show_popup(text)
            pass
        else:
            toast("you are safe")

    def mic_clicked(self):
        if self.recscreen.micbutton.source == "resources/icons/micoff.png":  # Turn mic on
            self.recscreen.micbutton.source = "resources/icons/micon.png"
            print("mic clicked")
            self.externally_stopped = False
            toast("started")
            th = threading.Thread(target=self.thread_for_rec())
            th.start()

        else:
            try:
                sd.stop()
                self.externally_stopped = True
                fs = 44100  # Sample rate
                write('recorded.wav', fs, self.myrecording)  # Save as WAV file in 16-bit format
                self.filename = "recorded.wav"
                toast("stopped")
            except:
                print("hello")
            self.recscreen.micbutton.source = "resources/icons/micoff.png"

    def loadfile(self, path, selection):

        # print(selection)
        self.filename = str(selection[0])
        self.fileloaderscreen_to_internalstoragescreen()

    def internalstoragescreen_to_mainscreen(self):
        self.screen_manager.transition.direction = 'right'
        self.screen_manager.current = 'mainscreen'

    def mainscreen_to_internalstoragescreen(self):
        self.screen_manager.transition.direction = 'left'
        self.screen_manager.current = 'internalstoragescreen'

    def mainscreen_to_recscreen(self):
        self.screen_manager.transition.direction = 'left'
        self.screen_manager.current = 'recscreen'

    def recscreen_to_mainscreen(self):
        self.screen_manager.transition.direction = 'right'
        self.screen_manager.current = 'mainscreen'

    def internalstoragescreen_to_fileloader(self):
        self.screen_manager.transition.direction = 'up'
        self.screen_manager.current = 'fileloaderscreen'

    def fileloaderscreen_to_internalstoragescreen(self):
        self.screen_manager.transition.direction = 'down'
        self.screen_manager.current = 'internalstoragescreen'

    def mainscreen_to_helpscreen(self):
        self.screen_manager.transition.direction = 'down'
        self.screen_manager.current = 'helpscreen'

    def mainscreen_to_teamscreen(self):
        self.screen_manager.transition.direction = 'down'
        self.screen_manager.current = 'teamscreen'

    def backforcommonscreens(self):
        self.screen_manager.transition.direction = 'up'
        self.screen_manager.current = 'mainscreen'

    def show_alert_dialog(self):
        if not self.dialog:
            self.dialog = MDDialog(
                text="Engine is currently Running!!",
                buttons=[
                    MDFlatButton(
                        text="Ok", text_color=self.theme_cls.primary_color, on_press=lambda x: self.dialog.dismiss(),
                    )
                ],

            )
        self.dialog.open()


if __name__ == '__main__':
    LabelBase.register(name='second', fn_regular='FFF_Tusj.ttf')
    LabelBase.register(name='first', fn_regular='Pacifico.ttf')

    uiApp().run()