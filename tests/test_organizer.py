# test_organizer.py
from src.core.organizer import DatasetOrganizer

test_folder = "/home/user/test_data/mix_data"

organizer = DatasetOrganizer(test_folder)
organizer.auto_organize()