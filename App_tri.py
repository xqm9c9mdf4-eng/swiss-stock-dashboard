import os
import pytesseract
from PIL import Image
import cv2
import re
import shutil  


# ================================
# CONFIGURATION TESSERACT WINDOWS
# ================================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ================================
# OCR
# ================================
def extract_text_from_image(image_path: str) -> str:
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {image_path}")

    # Amélioration pour OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    temp = "temp_ocr.png"
    cv2.imwrite(temp, gray)

    text = pytesseract.image_to_string(Image.open(temp), lang="fra+eng")
    os.remove(temp)

    return text


# ================================
# CLASSIFICATION
# ================================
def classify_document(text: str) -> str:
    t = text.lower()

    # Assurance maladie
    if any(k in t for k in [
        "assurance maladie", "lamal", "helsana", "CSS", "sanitas"
    ]):
        return "assurance_maladie"

    # Assurance mobilière
    if any(k in t for k in [
        "assurance", "mobilière"
    ]):
        return "assurance_mobiliere"

    # Factures
    if any(k in t for k in [
        "facture", "montant à payer", "échéance", "iban", "payable jusqu"
    ]):
        return "facture"

    # Impôts
    if any(k in t for k in [
        "impôt", "impots", "administration fiscale", "déclaration d'impôt", "taxes"
    ]):
        return "impots"

    # Banque
    if any(k in t for k in [
        "relevé de compte", "extrait de compte", "bénéficiaire", "virement", "banque"
    ]):
        return "banque"

    return "inconnu"


# ================================
# NOM DE FICHIER
# ================================
def propose_filename(category: str, text: str) -> str:
    match = re.search(r"(\d{2}\.\d{2}\.\d{4})", text)

    if match:
        date = match.group(1).replace(".", "-")
    else:
        date = "sans_date"

    return f"{date}_{category}.pdf"


# ================================
# PROGRAMME PRINCIPAL
# ================================
def main():
    # Mets ici le chemin exact vers ton image
    image_path = r"photo1.png"

    print("=== SCRIPT VERSION 2 ===")
    print("Analyse du document :", image_path)

    # Petit check pour être sûr que le fichier existe
    if not os.path.exists(image_path):
        print("⚠ Le fichier n'existe pas, vérifie le chemin / le nom.")
        return

    # OCR
    text = extract_text_from_image(image_path)

    print("\n=== TEXTE OCR (début) ===")
    print(text[:600])
    print("========================\n")

    # Classification
    category = classify_document(text)
    print("Catégorie détectée :", category)

    # Nom du fichier
    filename = propose_filename(category, text)
    print("Nom de fichier proposé (base) :", filename)

    # Dossier de classement
    base_folder = r"C:\Users\Franc\OneDrive\Documents\03_Programation\App\ClassementDocs"
    category_folder = os.path.join(base_folder, category)
    os.makedirs(category_folder, exist_ok=True)

    # Conserver l’extension originale
    name_without_ext, _ = os.path.splitext(filename)
    _, image_ext = os.path.splitext(image_path)
    final_name = name_without_ext + image_ext

    dest_path = os.path.join(category_folder, final_name)

    # Copier la photo dans le dossier de classement
    shutil.copy2(image_path, dest_path)
    # Si tu préfères déplacer :
    # shutil.move(image_path, dest_path)

    print("Fichier enregistré dans :", dest_path)
    print("\n✅ Classification terminée !")


# LANCEMENT DU SCRIPT
if __name__ == "__main__":
    main()
