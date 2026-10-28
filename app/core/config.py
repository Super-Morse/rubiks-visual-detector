from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Rubik's Visual Detector"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
