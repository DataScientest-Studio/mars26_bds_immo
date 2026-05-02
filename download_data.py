import urllib.request
import zipfile
import os
import pandas as pd
from pathlib import Path

DATA_URL1 = "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20260405-002306/valeursfoncieres-2024.txt.zip"
DATA_URL2 = "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20260405-002321/valeursfoncieres-2025.txt.zip"
DATA_DIR = "data/raw/"
ZIP_PATH1 = os.path.join(DATA_DIR, "download1.zip")
ZIP_PATH2 = os.path.join(DATA_DIR, "download2.zip")
DATA_DIRObjet = Path("data/raw/")
all_data = []

def download_and_extract():
    """Télécharge et extrait les données"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"Téléchargement de {DATA_URL1}...")
    print(f"Téléchargement de {DATA_URL2}...")
    urllib.request.urlretrieve(DATA_URL1, ZIP_PATH1)
    urllib.request.urlretrieve(DATA_URL2, ZIP_PATH2)
    
    print("Extraction en cours...")
    with zipfile.ZipFile(ZIP_PATH1, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    with zipfile.ZipFile(ZIP_PATH2, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    

    for txt_file in DATA_DIRObjet.rglob("*.txt"):
        data = pd.read_csv(txt_file, sep="|")
        csv_file = txt_file.with_suffix(".csv")
        all_data.append(data)
        #data.to_csv(csv_file, index=False)
        txt_file.unlink()
    merged_data = pd.concat(all_data, ignore_index=True)


    # Sauvegarder
    merged_data.to_csv(DATA_DIR+"valeurs_foncieres_2024_2025.csv", index=False)
    
    print("✓ Données téléchargées et extraites!")
    os.remove(ZIP_PATH1)  # Supprime le zip après extraction
    os.remove(ZIP_PATH2)  # Supprime le zip après extraction

if __name__ == "__main__":
    download_and_extract()