from pymongo import MongoClient
import certifi
import os

ca = certifi.where()

client = MongoClient('mongodb', int(os.getenv("MONGO_PORT")))

db = client.Roof_Surface_Website

collection_user = db["User"]
collection_factory = db["Factory"]
collection_building = db["Building"]
collection_Image = db["Image"]
collection_DefectLocation = db["DefectLocation"]
collection_Defect = db["Defect"]
collection_Permission = db["Permission"]
collection_history = db["History"]
collection_log = db["Log"]