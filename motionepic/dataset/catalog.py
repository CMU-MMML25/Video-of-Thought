import os


class DatasetCatalog:
    def __init__(self):

        self.webvid_enc = {
            "target": "motionepic.dataset.base_dataset.LazySupervisedDataset",
            "params": dict(
                data_path="./data/webvid/webvid.json",
                video_folder="",
            ),
        }
        
        # Add STAR dataset configuration
        self.STAR = {
            "target": "motionepic.dataset.star_dataset.STARDataset",
            "params": dict(
                data_path="../STAR/data/STAR_train.json",
                video_folder="../STAR/data/Charades_v1_480",
            ),
        }