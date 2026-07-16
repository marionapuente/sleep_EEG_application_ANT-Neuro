# Import packages
import matplotlib
matplotlib.use("Agg")
import os
from shiny import App
from app_ui import app_ui
from server import server

# Ensure correct working directory
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)

# Run the app
app = App(app_ui, server)
