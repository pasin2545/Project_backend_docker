from fastapi import FastAPI
from routes.route import router
from fastapi.middleware.cors import CORSMiddleware
import os

os.makedirs("/app/data", exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

app.include_router(router)