"""Backend configuration using pydantic-settings."""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # App
    APP_NAME: str = "CDU Optimizer"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # DWSIM
    ## This path is to be updated by users to point to their local DWSIM installation. The default here is for my installation.
    DWSIM_PATH: str = r"C:\Users\sigma\AppData\Local\DWSIM"
    FLOWSHEET_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Sim_models", "main_sim.dwxmz"
    )

    # Firebase
    FIREBASE_CREDENTIALS_PATH: Optional[str] = None
    FIREBASE_PROJECT_ID: Optional[str] = None

    # Google Gemini (for AI Agent)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # RL Training
    RL_LEARNING_RATE: float = 3e-4
    RL_BATCH_SIZE: int = 256
    RL_BUFFER_SIZE: int = 100_000
    RL_GAMMA: float = 0.99
    RL_TAU: float = 0.005
    RL_TRAINING_STEPS: int = 50_000
    RL_CHECKPOINT_DIR: str = "checkpoints"

    # Product names for the CDU + VDU
    PRODUCT_NAMES: list[str] = [
        "Uncondensed_Gas", "USN", "HN", "SKO", "LD", "HD",
        "Vac_Diesel", "VGO", "Slop_Cut", "Vac_Residue",
    ]

    # Safety limits (temperatures in °C, pressures in kPa)
    MAX_COLUMN_TEMP: float = 400.0
    MIN_COLUMN_TEMP: float = 30.0
    MAX_COLUMN_PRESSURE: float = 300.0
    MIN_COLUMN_PRESSURE: float = 100.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # silently ignore .env keys not declared in Settings (e.g. frontend Firebase vars)


settings = Settings()
