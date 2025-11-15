import os
from dataclasses import dataclass

@dataclass
class CFG:
    data_folder:str
    model_checkpoint_folder:str
    model_save_folder:str
