import os
from pathlib import Path
from dotenv import load_dotenv

def load_app_env() -> str:
    """
    Loads environment files based on the current APP_ENV.
    Priority:
      1. .env.{APP_ENV} if APP_ENV is set (e.g. .env.production, .env.development)
      2. Fallback to standard .env
    """
    app_env = os.environ.get("APP_ENV", "development").lower()
    root_dir = Path(__file__).resolve().parent.parent.parent
    
    env_file = root_dir / f".env.{app_env}"
    if env_file.exists():
        load_dotenv(str(env_file))
        print(f"[INFO] Loaded environment configuration from: {env_file}")
    else:
        print(f"[INFO] Environment config {env_file} not found. Using system env or standard .env.")
        
    fallback_env = root_dir / ".env"
    if fallback_env.exists():
        load_dotenv(str(fallback_env))
        
    os.environ["APP_ENV"] = app_env
    return app_env
