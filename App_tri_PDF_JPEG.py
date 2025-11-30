import os
import pytesseract
from PIL import Image
import cv2
import re
import shutil
import fitz  # PyMuPDF pour lire les PDF

# ================================
# CONFIGURATION TESSERACT WINDOWS
# ================================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ================================
# OCR SUR IMAGE (PNG/JPG…)
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
# OCR SUR PDF
# ================================
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Convertit chaque page du PDF en image, fait l'OCR, et concatène le texte.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    doc = fitz.open(pdf_path)
    all_text = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        # Rendu de la page en image (300 dpi pour un meilleur OCR)
        pix = page.get_pixmap(dpi=300)
        temp_img = f"temp_pdf_page_{page_index}.png"
        pix.save(temp_img)

        text_page = extract_text_from_image(temp_img)
        all_text.append(text_page)

        os.remove(temp_img)

    doc.close()
    return "\n\n".join(all_text)


# ================================
# OCR GÉNÉRIQUE (IMAGE OU PDF)
# ================================
def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(path)
    else:
        # On considère tout le reste comme image (png, jpg, jpeg, etc.)
        return extract_text_from_image(path)


# ================================
# CLASSIFICATION
# ================================
def classify_document(text: str) -> str:
    t = text.lower()

    # Assurance maladie
    if any(k in t for k in [
        "assurance maladie", "lamal", "helsana", "css assurance", "sanitas"
    ]):
        return "assurance_maladie"

    # Assurance mobilière
    if any(k in t for k in [
        "assurance ménage", "assurance mobilière", "Mobilière", "rc", "véhicules"
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
    # 👉 ICI tu mets soit une photo, soit un PDF

    # Exemple : pour tester un PDF
    # file_path = r"doc.pdf"

    # Exemple : pour tester une image
    file_path = r"photo.png"

    print("=== SCRIPT PDF + IMAGES ===")
    print("Analyse du fichier :", file_path)

    # Vérifier que le fichier existe
    if not os.path.exists(file_path):
        print("⚠ Le fichier n'existe pas, vérifie le chemin / le nom.")
        return

    # OCR (automatique selon l'extension)
    text = extract_text_from_file(file_path)

    print("\n=== TEXTE OCR (début) ===")
    print(text[:600])
    print("========================\n")

    # Classification
    category = classify_document(text)
    print("Catégorie détectée :", category)

    # Nom de fichier "logique"
    filename = propose_filename(category, text)
    print("Nom de fichier proposé (base) :", filename)

    # Dossier de classement
    base_folder = r"C:\Users\Franc\OneDrive\Documents\03_Programation\App\ClassementDocs"
    category_folder = os.path.join(base_folder, category)
    os.makedirs(category_folder, exist_ok=True)

    # Conserver l’extension originale (pdf ou image)
    name_without_ext, _ = os.path.splitext(filename)
    _, file_ext = os.path.splitext(file_path)
    final_name = name_without_ext + file_ext

    dest_path = os.path.join(category_folder, final_name)

    # Copier l’original vers le dossier de classement
    shutil.copy2(file_path, dest_path)
    # Si tu veux le déplacer au lieu de copier :
    # shutil.move(file_path, dest_path)

    print("Fichier enregistré dans :", dest_path)
    print("\n✅ Classification terminée !")


# LANCEMENT DU SCRIPT
if __name__ == "__main__":
    main()

