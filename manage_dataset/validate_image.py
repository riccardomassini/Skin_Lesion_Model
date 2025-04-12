from PIL import Image, UnidentifiedImageError


def is_image_valid(image_path):
    try:
        image = Image.open(image_path)
        image.verify()
        return True

    except (UnidentifiedImageError, IOError) as e:
        print(f"Errore con l'immagine: {image_path}. Dettagli: {e}")
        return False