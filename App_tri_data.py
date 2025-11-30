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
# PARAMÈTRES GLOBAUX
# ================================
INPUT_FOLDER = r"C:\Users\Franc\OneDrive\Documents\03_Programation\App\A_classer"
BASE_FOLDER = r"C:\Users\Franc\OneDrive\Documents\03_Programation\App\ClassementDocs"
ALLOWED_EXT = {".pdf", ".png", ".jpg", ".jpeg"}


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

    try:
        text = pytesseract.image_to_string(Image.open(temp), lang="fra+eng")
    finally:
        if os.path.exists(temp):
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

        try:
            text_page = extract_text_from_image(temp_img)
            all_text.append(text_page)
        finally:
            if os.path.exists(temp_img):
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
# CLASSIFICATION DU DOCUMENT
# ================================
def classify_document(text: str):
    """
    Retourne (categorie, raison) où raison = mot-clé déclencheur
    ou message générique si rien trouvé.
    """
    t = text.lower()

    rules = {
        "assurance_maladie": [
            "assurance maladie", "lamal", "helsana", "css assurance", "sanitas"
        ],
        "assurance_mobiliere": [
            "assurance ménage", "assurance menage",
            "assurance mobilière", "assurance mobiliere",
            "mobilière", "mobiliere", "véhicules", "vehicules"
        ],
        "facture": [
            "facture", "montant à payer", "montant a payer", "payable jusqu"
        ],
        "impots": [
            "impôt", "impots", "impot",
            "administration fiscale", "déclaration d'impôt",
            "declaration d'impot", "taxes", "numéro cantonal"
        ],
        "banque": [
            "relevé de compte", "releve de compte", "extrait de compte",
            "bénéficiaire", "beneficiaire", "virement",
            "iban", "banque", "bcv", "ubs", "raiffeisen",
            "postfinance", "attestation fiscale", "Epargne 3", "BCV"
        ]
    }

    for category, keywords in rules.items():
        for kw in keywords:
            if kw in t:
                return category, f"mot-clé détecté : '{kw}'"

    return "inconnu", "aucun mot-clé connu détecté"


# ================================
# DÉTECTION DU NOM DE LA BANQUE
# ================================
def detect_bank_name(text: str) -> str:
    """
    Essaie de détecter le nom de la banque dans le texte.
    Renvoie une version propre pour le nom de fichier (sans espaces).
    """
    t = text.lower()

    banks = {
        "BCV": ["bcv", "banque cantonale vaudoise"],
        "UBS": ["ubs"],
        "CREDIT_SUISSE": ["credit suisse", "crédit suisse"],
        "RAIFFEISEN": ["raiffeisen"],
        "POSTFINANCE": ["postfinance", "post finance"],
        "MIGROS_BANK": ["banque migros", "migros bank"],
        "REVOLUT": ["revolut"],
        "NEON": ["neon"],
        "YUH": ["yuh"],
        "CLAIRE": ["claire"],
    }

    for label, keywords in banks.items():
        for kw in keywords:
            if kw in t:
                return label

    # Si aucune banque spécifique trouvée
    return "BANQUE"


# ================================
# NOM DE FICHIER
# ================================
def propose_filename(category: str, text: str, bank_name: str | None = None) -> str:
    """
    Génère un nom de fichier basé sur :
    - la date trouvée dans le document
    - la catégorie
    - et si catégorie = banque -> la banque détectée
    """
    # Recherche d'une date au format 01.02.2024 ou 01/02/2024 ou 01-02-2024
    match = re.search(r"(\d{2}[./-]\d{2}[./-]\d{4})", text)

    if match:
        date = match.group(1).replace(".", "-").replace("/", "-")
    else:
        date = "sans_date"

    # Cas spécial : documents bancaires
    if category == "banque":
        if not bank_name:
            bank_name = detect_bank_name(text)
        return f"{date}_{bank_name}.pdf"

    # Cas général
    return f"{date}_{category}.pdf"


# ================================
# CONVERSION EN PDF
# ================================
def save_as_pdf(src_path: str, dest_pdf_path: str):
    """
    Sauvegarde le document en PDF.
    - Si c'est déjà un PDF : copie simple.
    - Si c'est une image : convertit en PDF avec PIL.
    """
    ext = os.path.splitext(src_path)[1].lower()

    # Si c'est déjà un PDF -> copie
    if ext == ".pdf":
        shutil.copy2(src_path, dest_pdf_path)
        return

    # Sinon on suppose que c'est une image supportée par PIL
    img = Image.open(src_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(dest_pdf_path, "PDF", resolution=300.0)


# ================================
# TRAITER UN SEUL FICHIER
# ================================
def process_file(file_path: str, base_folder: str):
    """Traite un seul fichier (image ou PDF)."""

    print("\n=== SCRIPT PDF + IMAGES ===")
    print("Analyse du fichier :", file_path)

    if not os.path.exists(file_path):
        print("⚠ Le fichier n'existe pas, vérifie le chemin / le nom.")
        return

    # OCR (automatique selon l'extension)
    text = extract_text_from_file(file_path)

    print("\n=== TEXTE OCR (début) ===")
    print(text[:600])
    print("\n========================\n")

    # Classification
    category, reason = classify_document(text)
    print("Catégorie détectée :", category)
    print("Raison :", reason)

    # Nom de fichier "logique"
    bank_name = detect_bank_name(text) if category == "banque" else None
    filename = propose_filename(category, text, bank_name)
    print("Nom de fichier proposé (base) :", filename)

    # Dossier de classement
    category_folder = os.path.join(base_folder, category)
    os.makedirs(category_folder, exist_ok=True)

    # On génère UNIQUEMENT un PDF, pas de copie de l'original
    name_without_ext, _ = os.path.splitext(filename)  # filename finit déjà par .pdf
    dest_pdf_path = os.path.join(category_folder, name_without_ext + ".pdf")

    save_as_pdf(file_path, dest_pdf_path)
    print("PDF enregistré :", dest_pdf_path)
    print("✅ Fichier terminé.")


# ================================
# PROGRAMME PRINCIPAL : TRAITER TOUT UN DOSSIER
# ================================
def main():
    input_folder = INPUT_FOLDER
    base_folder = BASE_FOLDER
    allowed_ext = ALLOWED_EXT

    if not os.path.exists(input_folder):
        print("⚠ Le dossier d'entrée n'existe pas :", input_folder)
        return

    files = os.listdir(input_folder)
    if not files:
        print("⚠ Aucun fichier trouvé dans :", input_folder)
        return

    print(f"📂 {len(files)} élément(s) dans {input_folder}")

    for name in files:
        file_path = os.path.join(input_folder, name)

        # Ignorer les sous-dossiers
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(name)[1].lower()
        if ext not in allowed_ext:
            print(f"⏭ Fichier ignoré (extension non supportée) : {name}")
            continue

        print("\n===============================")
        print("➡ Nouveau fichier :", name)
        print("===============================")

        process_file(file_path, base_folder)


# LANCEMENT DU SCRIPT
if __name__ == "__main__":
    main()
