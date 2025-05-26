import requests
import os
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time
from PIL import Image
from io import BytesIO
import hashlib


class ImageScraper:
    def __init__(self, download_path="./images", min_size=(100, 100)):
        """
        Inizializza lo scraper per immagini

        Args:
            download_path: Path dove salvare le immagini
            min_size: Dimensione minima delle immagini (width, height)
        """
        self.download_path = download_path
        self.min_size = min_size
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Crea la directory se non esiste
        os.makedirs(download_path, exist_ok=True)

    def get_images_from_page(self, url):
        """
        Estrae tutti gli URL delle immagini da una pagina web
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            img_tags = soup.find_all('img')

            img_urls = []
            for img in img_tags:
                img_url = img.get('src') or img.get('data-src')
                if img_url:
                    # Converte URL relativi in assoluti
                    full_url = urljoin(url, img_url)
                    img_urls.append(full_url)

            return img_urls

        except Exception as e:
            print(f"Errore nell'estrazione delle immagini da {url}: {e}")
            return []

    def is_valid_image(self, img_content):
        """
        Verifica se il contenuto è un'immagine valida e rispetta i requisiti di dimensione
        """
        try:
            img = Image.open(BytesIO(img_content))
            width, height = img.size

            # Controlla se l'immagine rispetta le dimensioni minime
            if width >= self.min_size[0] and height >= self.min_size[1]:
                return True, img.format.lower()
            return False, None

        except Exception:
            return False, None

    def download_image(self, img_url, filename):
        """
        Scarica una singola immagine
        """
        try:
            response = requests.get(img_url, headers=self.headers, timeout=15)
            response.raise_for_status()

            # Verifica se è un'immagine valida
            is_valid, img_format = self.is_valid_image(response.content)
            if not is_valid:
                return False

            # Crea un nome file unico basato sull'hash del contenuto
            img_hash = hashlib.md5(response.content).hexdigest()[:10]
            file_extension = img_format if img_format else 'jpg'
            full_filename = f"{filename}_{img_hash}.{file_extension}"
            file_path = os.path.join(self.download_path, full_filename)

            # Evita duplicati
            if os.path.exists(file_path):
                return False

            # Salva l'immagine
            with open(file_path, 'wb') as f:
                f.write(response.content)

            print(f"✓ Scaricata: {full_filename}")
            return True

        except Exception as e:
            print(f"✗ Errore nel download di {img_url}: {e}")
            return False

    def scrape_images_from_search(self, query, num_images, search_engine="unsplash"):
        """
        Cerca e scarica immagini da un motore di ricerca

        Args:
            query: Termine di ricerca
            num_images: Numero di immagini da scaricare
            search_engine: Motore di ricerca da utilizzare
        """
        if search_engine == "unsplash":
            search_url = f"https://unsplash.com/s/photos/{query.replace(' ', '-')}"
        elif search_engine == "pexels":
            search_url = f"https://www.pexels.com/search/{query.replace(' ', '%20')}/"
        else:
            print("Motore di ricerca non supportato")
            return

        print(f"Cercando immagini per: {query}")
        print(f"URL di ricerca: {search_url}")

        img_urls = self.get_images_from_page(search_url)

        if not img_urls:
            print("Nessuna immagine trovata nella pagina")
            return

        print(f"Trovate {len(img_urls)} immagini potenziali")

        downloaded = 0
        for i, img_url in enumerate(img_urls):
            if downloaded >= num_images:
                break

            print(f"Scaricando immagine {downloaded + 1}/{num_images}...")

            if self.download_image(img_url, f"{query.replace(' ', '_')}_{i + 1}"):
                downloaded += 1

            # Pausa per evitare di sovraccaricare il server
            time.sleep(1)

        print(f"\n✓ Scaricate {downloaded} immagini in '{self.download_path}'")

    def scrape_images_from_urls(self, urls, num_images_per_url):
        """
        Scarica immagini da una lista di URL specificati

        Args:
            urls: Lista di URL da cui scaricare immagini
            num_images_per_url: Numero di immagini da scaricare per ogni URL
        """
        total_downloaded = 0

        for url in urls:
            print(f"\nEstraendo immagini da: {url}")
            img_urls = self.get_images_from_page(url)

            downloaded_from_url = 0
            for i, img_url in enumerate(img_urls):
                if downloaded_from_url >= num_images_per_url:
                    break

                domain = urlparse(url).netloc.replace('www.', '')
                filename = f"{domain}_{i + 1}"

                if self.download_image(img_url, filename):
                    downloaded_from_url += 1
                    total_downloaded += 1

                time.sleep(0.5)

            print(f"Scaricate {downloaded_from_url} immagini da {url}")

        print(f"\n✓ Totale immagini scaricate: {total_downloaded}")


# Esempio di utilizzo
if __name__ == "__main__":
    # Inizializza lo scraper
    scraper = ImageScraper(
        download_path="./images",
        min_size=(200, 200)  # Immagini minimo 200x200 pixel
    )

    # Metodo 1: Cerca immagini per query
    print("=== RICERCA PER QUERY ===")
    scraper.scrape_images_from_search(
        query="nature landscape",
        num_images=10,
        search_engine="unsplash"
    )

    # Metodo 2: Scarica da URL specifici
    print("\n=== SCARICA DA URL SPECIFICI ===")
    urls = [
        "https://example-photography-site.com",
        "https://another-image-site.com"
    ]
    # scraper.scrape_images_from_urls(urls, num_images_per_url=5)

    print("\nScraping completato!")