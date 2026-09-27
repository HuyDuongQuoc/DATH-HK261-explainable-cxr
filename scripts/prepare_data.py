from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

# get username and password for kaggle authentication
load_dotenv()

# download dataset
api = KaggleApi()
api.authenticate()

dataset = "sunghyunjun/vinbigdata-1024-jpg-dataset"
save_dir = "./data/raw"

api.dataset_download_files(
    dataset=dataset,
    path=save_dir,
    unzip=True,
    quiet=False
)