import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src import api as vg
from PIL import Image as PIL_Image
import requests
from io import StringIO

class VisualGenomeDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_data = self._load_json('image_data.json')
        self.region_descriptions = self._load_json('region_descriptions.json')
        self.objects = self._load_json('objects.json')
        self.attributes = self._load_json('attributes.json')
        self.relationships = self._load_json('relationships.json')

    def _load_json(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data

    def get_image_data(self):
        return self.image_data

    def get_region_descriptions(self):
        return self.region_descriptions

    def get_objects(self):
        return self.objects

    def get_attributes(self):
        return self.attributes

    def get_relationships(self):
        return self.relationships

# Example usage:
# data_loader = VisualGenomeDataLoader('/path/to/visual_genome')
# image_data = data_loader.get_image_data()
# print(image_data)