import logging

from dotenv import load_dotenv

from src.cli.app import main

load_dotenv(".env.local")
load_dotenv(".env")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    main()
