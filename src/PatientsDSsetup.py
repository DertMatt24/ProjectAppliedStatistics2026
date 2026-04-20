import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from loader.eeg_recording import EEGLoader
from loader.patients import PatientsCSVLoader
from scipy.stats import kurtosis, skew
import numpy as np
from pathlib import Path

class PatientsDSsetup:
    def __init__(self):
        raise SyntaxError("Cannot instantiate PatientsDSsetup, it is an utility static class!")

    @staticmethod
    def __read_directory():
        current = Path(__file__).resolve().parent

        if current.name == "src":
            file_path = current / "noCommitFile" / "PatientsDirectory.txt"
        else:
            file_path = current / "src" / "noCommitFile" / "PatientsDirectory.txt"

        if not file_path.exists():
            print(f"DEBUG: Il file NON è stato trovato in: {file_path}")
            # Se vedi che manca un pezzo di percorso, aggiungi .parent a current_dir

        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    @staticmethod
    def load_dataframe():
        file_path = PatientsDSsetup.__read_directory()
        patient = PatientsCSVLoader.load_dataframe(file_path)

        return patient

    @staticmethod
    def add_BMI(patient):
        patient['BMI'] = patient['weight'] / ((patient['height'] / 100) ** 2)

        # 2. Define the conditions and the labels
        conditions = [
            (patient['BMI'] < 18.5),
            (patient['BMI'] >= 18.5) & (patient['BMI'] < 25),
            (patient['BMI'] >= 25) & (patient['BMI'] < 30),
            (patient['BMI'] >= 30)
        ]

        choices = ['underweight', 'normal', 'overweight', 'obese']

        # 3. Create the 'obesity' variable
        patient['obesity'] = np.select(conditions, choices, default='unknown')

        return patient