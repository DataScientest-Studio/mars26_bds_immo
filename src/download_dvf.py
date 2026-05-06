import os
import gdown

DATA_DIR = "data/raw"
FILE_ID = "1QrskxqmPXCf3Uw_7vx9Sc8VlQfyIc_Ac"
OUTPUT_FILE = os.path.join(DATA_DIR, "dvf_final_2020_2025.csv")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    print("Téléchargement du fichier DVF depuis Google Drive...")
    gdown.download(url, OUTPUT_FILE, quiet=False)

    print(f"Fichier téléchargé : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
