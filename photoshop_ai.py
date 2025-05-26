import os
import json
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import random
import hashlib


class ImageAnalyzer:
    """Analizza le caratteristiche delle immagini con maggiore precisione"""

    def __init__(self):
        self.label_encoder = LabelEncoder()

    def extract_features(self, image_path):
        """Estrae features complete dall'immagine"""
        try:
            # Carica immagine
            image = cv2.imread(image_path)
            if image is None:
                return None

            # Converti in diversi spazi colore per analisi più approfondita
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

            # Features base migliorati
            features = {
                'brightness': np.mean(gray),
                'contrast': np.std(gray),
                'aspect_ratio': image.shape[1] / image.shape[0],
                'resolution': image.shape[0] * image.shape[1],
                'dominant_color': self._get_dominant_color(image),
                'has_faces': self._detect_faces(image),
                'image_type': self._classify_image_type(image),

                # Nuove features per migliore analisi
                'color_temperature': self._estimate_color_temperature(image),
                'saturation_level': np.mean(hsv[:, :, 1]),
                'exposure_level': self._estimate_exposure(gray),
                'shadow_detail': self._analyze_shadows(gray),
                'highlight_detail': self._analyze_highlights(gray),
                'color_cast': self._detect_color_cast(lab),
                'noise_level': self._estimate_noise(gray),
                'sharpness': self._estimate_sharpness(gray),
                'dynamic_range': self._calculate_dynamic_range(gray)
            }

            # Aggiungi metadati EXIF se disponibili
            exif_data = self._extract_exif(image_path)
            features.update(exif_data)

            return features

        except Exception as e:
            print(f"Errore nell'analisi di {image_path}: {e}")
            return None

    def _get_dominant_color(self, image):
        """Trova il colore dominante con K-means migliorato"""
        data = image.reshape((-1, 3))
        data = np.float32(data)

        # K-means per trovare 3 colori dominanti
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Restituisce il colore più dominante
        unique, counts = np.unique(labels, return_counts=True)
        dominant_idx = unique[np.argmax(counts)]
        return centers[dominant_idx].tolist()

    def _detect_faces(self, image):
        """Rileva volti con maggiore precisione"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        return len(faces)

    def _classify_image_type(self, image):
        """Classifica il tipo di immagine con più categorie"""
        height, width = image.shape[:2]
        aspect_ratio = width / height

        if aspect_ratio > 1.8:
            return "panoramic"
        elif aspect_ratio > 1.3:
            return "landscape"
        elif aspect_ratio < 0.7:
            return "portrait"
        else:
            return "square"

    def _estimate_color_temperature(self, image):
        """Stima la temperatura colore dell'immagine"""
        # Converti in RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Calcola il rapporto rosso/blu per stimare la temperatura
        r_mean = np.mean(rgb[:, :, 0])
        b_mean = np.mean(rgb[:, :, 2])

        if b_mean == 0:
            b_mean = 1

        ratio = r_mean / b_mean

        # Mappa approssimativa del rapporto alla temperatura Kelvin
        if ratio > 1.2:
            return "warm"  # >3200K
        elif ratio < 0.8:
            return "cool"  # <5600K
        else:
            return "neutral"  # 3200-5600K

    def _estimate_exposure(self, gray):
        """Stima il livello di esposizione"""
        mean_brightness = np.mean(gray)

        if mean_brightness < 60:
            return "underexposed"
        elif mean_brightness > 200:
            return "overexposed"
        else:
            return "normal"

    def _analyze_shadows(self, gray):
        """Analizza i dettagli nelle ombre"""
        shadow_pixels = gray[gray < 60]
        if len(shadow_pixels) == 0:
            return "no_shadows"

        shadow_detail = np.std(shadow_pixels)

        if shadow_detail < 5:
            return "blocked_shadows"
        elif shadow_detail > 15:
            return "good_shadow_detail"
        else:
            return "moderate_shadow_detail"

    def _analyze_highlights(self, gray):
        """Analizza i dettagli nelle alte luci"""
        highlight_pixels = gray[gray > 200]
        if len(highlight_pixels) == 0:
            return "no_highlights"

        highlight_detail = np.std(highlight_pixels)

        if highlight_detail < 5:
            return "blown_highlights"
        elif highlight_detail > 15:
            return "good_highlight_detail"
        else:
            return "moderate_highlight_detail"

    def _detect_color_cast(self, lab):
        """Rileva dominanti di colore indesiderate"""
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]

        a_mean = np.mean(a_channel) - 128
        b_mean = np.mean(b_channel) - 128

        if abs(a_mean) > 5 or abs(b_mean) > 5:
            if a_mean > 5:
                return "magenta_cast"
            elif a_mean < -5:
                return "green_cast"
            elif b_mean > 5:
                return "yellow_cast"
            elif b_mean < -5:
                return "blue_cast"

        return "no_cast"

    def _estimate_noise(self, gray):
        """Stima il livello di rumore nell'immagine"""
        # Usa il filtro Laplaciano per rilevare il rumore
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = np.var(laplacian)

        if noise_level > 1000:
            return "high_noise"
        elif noise_level > 500:
            return "moderate_noise"
        else:
            return "low_noise"

    def _estimate_sharpness(self, gray):
        """Stima la nitidezza dell'immagine"""
        # Usa il gradiente per misurare la nitidezza
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sharpness = np.mean(np.sqrt(sobelx ** 2 + sobely ** 2))

        if sharpness > 30:
            return "sharp"
        elif sharpness > 15:
            return "moderate_sharp"
        else:
            return "soft"

    def _calculate_dynamic_range(self, gray):
        """Calcola la gamma dinamica dell'immagine"""
        return int(np.max(gray)) - int(np.min(gray))

    def _extract_exif(self, image_path):
        """Estrae dati EXIF dall'immagine"""
        try:
            image = Image.open(image_path)
            exifdata = image.getexif()

            exif_dict = {}
            for tag_id in exifdata:
                tag = TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                exif_dict[tag] = data

            # Estrai solo i dati rilevanti
            relevant_exif = {
                'camera_make': exif_dict.get('Make', ''),
                'camera_model': exif_dict.get('Model', ''),
                'iso': exif_dict.get('ISOSpeedRatings', 0),
                'focal_length': exif_dict.get('FocalLength', 0),
                'aperture': exif_dict.get('FNumber', 0),
                'shutter_speed': exif_dict.get('ExposureTime', 0)
            }

            return relevant_exif

        except Exception:
            return {}


class AdjustmentAdvisor:
    """Calcola valori di regolazione più intelligenti e realistici"""

    def __init__(self):
        # Regole di regolazione basate sulle condizioni dell'immagine
        self.adjustment_rules = {
            'exposure': {
                'underexposed': (0.3, 1.5),
                'overexposed': (-1.5, -0.3),
                'normal': (-0.2, 0.2)
            },
            'shadows': {
                'blocked_shadows': (30, 80),
                'good_shadow_detail': (-10, 10),
                'moderate_shadow_detail': (10, 40),
                'no_shadows': (0, 0)
            },
            'highlights': {
                'blown_highlights': (-80, -30),
                'good_highlight_detail': (-10, 10),
                'moderate_highlight_detail': (-40, -10),
                'no_highlights': (0, 0)
            },
            'white_balance': {
                'warm': (-500, -100),
                'cool': (100, 500),
                'neutral': (-50, 50)
            },
            'vibrance': {
                'high_noise': (0, 20),
                'moderate_noise': (10, 40),
                'low_noise': (20, 60)
            }
        }

    def calculate_adjustment(self, action, image_features):
        """Calcola regolazioni intelligenti basate sulle features dell'immagine"""
        recommendation = {
            'action': action,
            'parameter': action,
            'value': 0,
            'direction': 'neutral',
            'intensity': 'medium',
            'reason': ''
        }

        if action == 'exposure':
            exposure_level = image_features.get('exposure_level', 'normal')
            value_range = self.adjustment_rules['exposure'].get(exposure_level, (-0.2, 0.2))
            recommendation['value'] = random.uniform(*value_range)
            recommendation['reason'] = f"Immagine rilevata come {exposure_level}"

        elif action == 'shadows':
            shadow_detail = image_features.get('shadow_detail', 'moderate_shadow_detail')
            value_range = self.adjustment_rules['shadows'].get(shadow_detail, (0, 20))
            recommendation['value'] = random.uniform(*value_range)
            recommendation['reason'] = f"Dettaglio ombre: {shadow_detail}"

        elif action == 'highlights':
            highlight_detail = image_features.get('highlight_detail', 'moderate_highlight_detail')
            value_range = self.adjustment_rules['highlights'].get(highlight_detail, (-20, 0))
            recommendation['value'] = random.uniform(*value_range)
            recommendation['reason'] = f"Dettaglio alte luci: {highlight_detail}"

        elif action == 'white_balance':
            color_temp = image_features.get('color_temperature', 'neutral')
            if color_temp in self.adjustment_rules['white_balance']:
                value_range = self.adjustment_rules['white_balance'][color_temp]
                recommendation['value'] = 5500 + random.uniform(*value_range)
            else:
                recommendation['value'] = 5500
            recommendation['reason'] = f"Temperatura colore rilevata: {color_temp}"

        elif action == 'vibrance':
            saturation = image_features.get('saturation_level', 50)
            if saturation < 30:
                recommendation['value'] = random.uniform(20, 50)
                recommendation['reason'] = "Saturazione bassa rilevata"
            elif saturation > 80:
                recommendation['value'] = random.uniform(-30, -10)
                recommendation['reason'] = "Saturazione alta rilevata"
            else:
                recommendation['value'] = random.uniform(-10, 20)
                recommendation['reason'] = "Saturazione normale"

        elif action == 'contrast':
            contrast = image_features.get('contrast', 40)
            if contrast < 25:
                recommendation['value'] = random.uniform(10, 40)
                recommendation['reason'] = "Contrasto basso rilevato"
            elif contrast > 70:
                recommendation['value'] = random.uniform(-40, -10)
                recommendation['reason'] = "Contrasto alto rilevato"
            else:
                recommendation['value'] = random.uniform(-15, 15)
                recommendation['reason'] = "Contrasto normale"

        elif action == 'clarity':
            sharpness = image_features.get('sharpness', 'moderate_sharp')
            if sharpness == 'soft':
                recommendation['value'] = random.uniform(15, 40)
                recommendation['reason'] = "Immagine poco nitida"
            elif sharpness == 'sharp':
                recommendation['value'] = random.uniform(-10, 10)
                recommendation['reason'] = "Immagine già nitida"
            else:
                recommendation['value'] = random.uniform(5, 25)
                recommendation['reason'] = "Nitidezza moderata"

        else:
            # Valori casuali più realistici per altre azioni
            recommendation['value'] = random.uniform(-20, 20)
            recommendation['reason'] = "Regolazione generica"

        # Determina direzione e intensità
        recommendation['direction'] = 'increase' if recommendation['value'] > 0 else 'decrease'
        abs_value = abs(recommendation['value'])

        if abs_value < 10:
            recommendation['intensity'] = 'low'
        elif abs_value < 30:
            recommendation['intensity'] = 'medium'
        else:
            recommendation['intensity'] = 'high'

        return recommendation


class WorkflowGenerator:
    """Genera workflow più intelligenti e variati"""

    def __init__(self):
        self.adjustment_advisor = AdjustmentAdvisor()

        # Workflow template più specifici
        self.workflow_templates = {
            'portrait_professional': [
                'exposure', 'white_balance', 'shadows', 'highlights', 'skin_tone',
                'eye_enhancement', 'teeth_whitening', 'background_blur'
            ],
            'portrait_artistic': [
                'exposure', 'contrast', 'vibrance', 'clarity', 'vignette',
                'color_grading', 'film_emulation'
            ],
            'landscape_natural': [
                'exposure', 'highlights', 'shadows', 'vibrance', 'clarity',
                'graduated_filter', 'polarizing_effect'
            ],
            'landscape_dramatic': [
                'exposure', 'contrast', 'structure', 'vibrance', 'clarity',
                'orton_effect', 'color_grading', 'vignette'
            ],
            'street_photography': [
                'exposure', 'contrast', 'clarity', 'grain', 'black_white_mix', 'vignette'
            ],
            'macro_photography': [
                'exposure', 'clarity', 'vibrance', 'sharpening', 'background_blur', 'vignette'
            ]
        }

        # Azioni condizionali più specifiche
        self.conditional_actions = {
            'underexposed': ['exposure', 'shadows', 'brightness'],
            'overexposed': ['exposure', 'highlights'],
            'blocked_shadows': ['shadows', 'black_point'],
            'blown_highlights': ['highlights', 'white_point'],
            'warm': ['white_balance', 'color_temperature'],
            'cool': ['white_balance', 'color_temperature'],
            'high_noise': ['noise_reduction', 'grain_reduction'],
            'soft': ['clarity', 'sharpening', 'structure'],
            'low_contrast': ['contrast', 'clarity'],
            'magenta_cast': ['tint', 'color_correction'],
            'green_cast': ['tint', 'color_correction']
        }

    def generate_intelligent_workflow(self, image_features):
        """Genera workflow basato su analisi intelligente delle features"""

        # Determina il tipo di workflow più appropriato
        workflow_type = self._determine_workflow_type(image_features)
        base_workflow = self.workflow_templates.get(workflow_type,
                                                    self.workflow_templates['portrait_artistic'])

        # Aggiungi azioni condizionali basate sui problemi rilevati
        additional_actions = self._get_conditional_actions(image_features)

        # Combina e ottimizza il workflow
        combined_workflow = self._combine_workflows(base_workflow, additional_actions)

        # Genera regolazioni dettagliate
        detailed_workflow = []
        for action in combined_workflow:
            adjustment = self.adjustment_advisor.calculate_adjustment(action, image_features)
            detailed_workflow.append({
                'action': adjustment['action'],
                'parameter': adjustment['parameter'],
                'value': adjustment['value'],
                'direction': adjustment['direction'],
                'intensity': adjustment['intensity'],
                'reason': adjustment['reason'],
                'description': self._get_action_description(action, adjustment)
            })

        return detailed_workflow

    def _determine_workflow_type(self, features):
        """Determina il tipo di workflow più appropriato"""

        # Logica decisionale migliorata
        if features.get('has_faces', 0) > 0:
            if features.get('noise_level') == 'low_noise' and features.get('sharpness') == 'sharp':
                return 'portrait_professional'
            else:
                return 'portrait_artistic'

        elif features.get('image_type') == 'landscape':
            if features.get('dynamic_range', 0) > 150:
                return 'landscape_dramatic'
            else:
                return 'landscape_natural'

        elif features.get('image_type') == 'panoramic':
            return 'landscape_natural'

        else:
            # Default per immagini generiche
            return 'street_photography'

    def _get_conditional_actions(self, features):
        """Determina azioni correttive necessarie"""
        actions = []

        # Controlla ogni possibile problema
        for condition, condition_actions in self.conditional_actions.items():
            if self._has_condition(features, condition):
                actions.extend(condition_actions)

        return list(set(actions))  # Rimuovi duplicati

    def _has_condition(self, features, condition):
        """Verifica se l'immagine ha una specifica condizione"""
        condition_map = {
            'underexposed': features.get('exposure_level') == 'underexposed',
            'overexposed': features.get('exposure_level') == 'overexposed',
            'blocked_shadows': features.get('shadow_detail') == 'blocked_shadows',
            'blown_highlights': features.get('highlight_detail') == 'blown_highlights',
            'warm': features.get('color_temperature') == 'warm',
            'cool': features.get('color_temperature') == 'cool',
            'high_noise': features.get('noise_level') == 'high_noise',
            'soft': features.get('sharpness') == 'soft',
            'low_contrast': features.get('contrast', 50) < 25,
            'magenta_cast': features.get('color_cast') == 'magenta_cast',
            'green_cast': features.get('color_cast') == 'green_cast'
        }

        return condition_map.get(condition, False)

    def _combine_workflows(self, base_workflow, additional_actions):
        """Combina workflow base con azioni aggiuntive in modo intelligente"""

        # Priorità delle azioni (le più importanti vanno per prime)
        priority_order = [
            'exposure', 'white_balance', 'highlights', 'shadows',
            'contrast', 'vibrance', 'clarity', 'saturation',
            'noise_reduction', 'sharpening', 'vignette'
        ]

        # Unisci tutte le azioni
        all_actions = list(set(base_workflow + additional_actions))

        # Ordina secondo le priorità
        ordered_actions = []
        for priority_action in priority_order:
            if priority_action in all_actions:
                ordered_actions.append(priority_action)
                all_actions.remove(priority_action)

        # Aggiungi le azioni rimanenti
        ordered_actions.extend(all_actions)

        # Limita la lunghezza (4-8 azioni)
        max_length = random.randint(4, 8)
        return ordered_actions[:max_length]

    def _get_action_description(self, action, adjustment):
        """Genera descrizioni più dettagliate delle azioni"""
        descriptions = {
            'exposure': f"Regola esposizione di {adjustment['value']:.1f} stop - {adjustment['reason']}",
            'contrast': f"{'Aumenta' if adjustment['value'] > 0 else 'Diminuisci'} contrasto del {abs(adjustment['value']):.0f}% - {adjustment['reason']}",
            'white_balance': f"Imposta bilanciamento bianco a {adjustment['value']:.0f}K - {adjustment['reason']}",
            'vibrance': f"{'Aumenta' if adjustment['value'] > 0 else 'Diminuisci'} vividezza del {abs(adjustment['value']):.0f}% - {adjustment['reason']}",
            'shadows': f"{'Apri' if adjustment['value'] > 0 else 'Chiudi'} ombre del {abs(adjustment['value']):.0f}% - {adjustment['reason']}",
            'highlights': f"{'Recupera' if adjustment['value'] < 0 else 'Aumenta'} alte luci del {abs(adjustment['value']):.0f}% - {adjustment['reason']}",
            'clarity': f"{'Aumenta' if adjustment['value'] > 0 else 'Diminuisci'} chiarezza del {abs(adjustment['value']):.0f}% - {adjustment['reason']}"
        }

        return descriptions.get(action,
                                f"Applica {action} ({adjustment['intensity']} intensity) - {adjustment['reason']}")


class WorkflowAI:
    """Classe principale migliorata"""

    def __init__(self):
        self.image_analyzer = ImageAnalyzer()
        self.workflow_tracker = WorkflowTracker()
        self.predictor = WorkflowPredictor()
        self.workflow_generator = WorkflowGenerator()

        # Carica modello esistente se disponibile
        try:
            self.predictor.load_model("workflow_model.pkl")
        except:
            print("Nessun modello esistente trovato. Verrà creato uno nuovo.")

    def analyze_and_suggest(self, image_path, current_actions=[]):
        """Analizza immagine e suggerisce workflow intelligente"""

        print(f"Analizzando {image_path}...")
        features = self.image_analyzer.extract_features(image_path)

        if features is None:
            return None

        # Genera workflow intelligente
        intelligent_workflow = self.workflow_generator.generate_intelligent_workflow(features)

        # Predici prossime azioni dal modello ML
        ml_predictions = self.predictor.predict_next_actions(features, current_actions)

        return {
            'image_features': features,
            'intelligent_workflow': intelligent_workflow,
            'ml_predictions': ml_predictions,
            'analysis_summary': self._generate_analysis_summary(features)
        }

    def _generate_analysis_summary(self, features):
        """Genera un riassunto dell'analisi per l'utente"""
        issues = []
        strengths = []

        # Identifica problemi
        if features.get('exposure_level') == 'underexposed':
            issues.append("Immagine sottoesposta")
        elif features.get('exposure_level') == 'overexposed':
            issues.append("Immagine sovraesposta")

        if features.get('shadow_detail') == 'blocked_shadows':
            issues.append("Ombre chiuse senza dettaglio")

        if features.get('highlight_detail') == 'blown_highlights':
            issues.append("Alte luci bruciate")

        if features.get('color_cast') != 'no_cast':
            issues.append(f"Dominante di colore: {features.get('color_cast')}")

        if features.get('noise_level') == 'high_noise':
            issues.append("Livello di rumore elevato")

        # Identifica punti di forza
        if features.get('sharpness') == 'sharp':
            strengths.append("Immagine ben nitida")

        if features.get('dynamic_range', 0) > 150:
            strengths.append("Buona gamma dinamica")

        if features.get('has_faces', 0) > 0:
            strengths.append(f"{features.get('has_faces')} volto/i rilevato/i")

        return {
            'issues': issues,
            'strengths': strengths,
            'overall_quality': 'good' if len(issues) <= 2 else 'needs_work'
        }

    def generate_diverse_samples(self, image_path, num_samples=10):
        """Genera campioni più diversificati per il training"""
        features = self.image_analyzer.extract_features(image_path)
        if not features:
            return

        print(f"Generando {num_samples} workflow diversificati per {image_path}")

        # Genera hash dell'immagine per evitare duplicati
        image_hash = self._generate_image_hash(image_path)

        for i in range(num_samples):
            # Genera workflow con variazioni
            workflow = self.workflow_generator.generate_intelligent_workflow(features)

            # Aggiungi variazioni casuali
            if random.random() > 0.7:  # 30% delle volte aggiungi azione casuale
                random_actions = ['grain', 'vignette', 'color_grading', 'film_emulation']
                workflow.append({
                    'action': random.choice(random_actions),
                    'description': f"Effetto artistico aggiunto"
                })

            action_names = [step['action'] for step in workflow]
            session_id = f"{image_hash}_{i}"

            self.workflow_tracker.log_workflow(image_path, action_names, features, session_id)

            print(f"  Workflow {i + 1}: {' → '.join(action_names[:4])}{'...' if len(action_names) > 4 else ''}")

    def _generate_image_hash(self, image_path):
        """Genera hash univoco per l'immagine"""
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return hashlib.md5(image_data).hexdigest()[:8]

    def show_detailed_analysis(self, image_path):
        """Mostra analisi dettagliata dell'immagine"""
        result = self.analyze_and_suggest(image_path)

        if not result:
            print("Impossibile analizzare l'immagine")
            return

        features = result['image_features']
        workflow = result['intelligent_workflow']
        summary = result['analysis_summary']

        print(f"\n=== ANALISI DETTAGLIATA: {os.path.basename(image_path)} ===")

        # Informazioni tecniche
        print(f"\n--- CARATTERISTICHE TECNICHE ---")
        print(f"Tipo: {features['image_type'].title()}")
        print(f"Risoluzione: {features['resolution']:,} px")
        print(f"Rapporto: {features['aspect_ratio']:.2f}")
        print(f"Luminosità: {features['brightness']:.1f}/255")
        print(f"Contrasto: {features['contrast']:.1f}")
        print(f"Gamma dinamica: {features['dynamic_range']}")
        print(f"Volti rilevati: {features['has_faces']}")

        # Valutazione qualità
        print(f"\n--- VALUTAZIONE QUALITÀ ---")
        print(f"Esposizione: {features['exposure_level'].title()}")
        print(f"Nitidezza: {features['sharpness'].title()}")
        print(f"Rumore: {features['noise_level'].title()}")
        print(f"Temperatura colore: {features['color_temperature'].title()}")
        print(f"Dettaglio ombre: {features['shadow_detail'].replace('_', ' ').title()}")
        print(f"Dettaglio alte luci: {features['highlight_detail'].replace('_', ' ').title()}")

        # Riassunto problemi e punti di forza
        print(f"\n--- RIASSUNTO ANALISI ---")
        if summary['issues']:
            print("Problemi rilevati:")
            for issue in summary['issues']:
                print(f"  • {issue}")

        if summary['strengths']:
            print("Punti di forza:")
            for strength in summary['strengths']:
                print(f"  • {strength}")

        print(f"Qualità generale: {summary['overall_quality'].replace('_', ' ').title()}")

        # Workflow suggerito
        print(f"\n--- WORKFLOW INTELLIGENTE SUGGERITO ---")
        for i, step in enumerate(workflow, 1):
            intensity_icon = "🔥" if step['intensity'] == 'high' else "⚡" if step['intensity'] == 'medium' else "💡"
            print(f"{i:2d}. {intensity_icon} {step['description']}")

        # Predizioni ML se disponibili
        if result['ml_predictions']:
            print(f"\n--- SUGGERIMENTI BASATI SUL TUO STILE ---")
            for pred in result['ml_predictions'][:3]:  # Top 3 predizioni
                print(f"  • {pred['action']} (confidenza: {pred['confidence']:.1%})")

    def train_and_save_model(self, min_samples=50):
        """Addestra e salva il modello se ci sono abbastanza dati"""
        if self.workflow_tracker.get_total_workflows() >= min_samples:
            print(f"Addestrando modello con {self.workflow_tracker.get_total_workflows()} campioni...")
            self.predictor.train_model()
            self.predictor.save_model("workflow_model.pkl")
            print("Modello salvato con successo!")
        else:
            print(f"Servono almeno {min_samples} campioni per addestrare il modello. "
                  f"Attualmente: {self.workflow_tracker.get_total_workflows()}")

class WorkflowTracker:
    """Traccia e memorizza i workflow degli utenti"""

    def __init__(self, db_name="workflows.db"):
        self.db_name = db_name
        self.init_database()

    def init_database(self):
        """Inizializza il database SQLite"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''
                CREATE TABLE IF NOT EXISTS workflows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    action_sequence TEXT NOT NULL,
                    image_features TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    workflow_type TEXT,
                    success_rating INTEGER DEFAULT 0
                )
            ''')

        cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    preference_key TEXT UNIQUE NOT NULL,
                    preference_value TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            ''')

        conn.commit()
        conn.close()

    def log_workflow(self, image_path, actions, features, session_id, workflow_type="auto", success_rating=0):
        """Registra un workflow nel database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()
        action_sequence = json.dumps(actions)
        image_features_json = json.dumps(features, default=str)

        cursor.execute('''
                INSERT INTO workflows 
                (image_path, timestamp, action_sequence, image_features, session_id, workflow_type, success_rating)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (image_path, timestamp, action_sequence, image_features_json, session_id, workflow_type,
                  success_rating))

        conn.commit()
        conn.close()

        print(f"Workflow registrato: {len(actions)} azioni per {os.path.basename(image_path)}")

    def get_user_workflows(self, limit=None):
        """Recupera i workflow dell'utente"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        query = "SELECT * FROM workflows ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        workflows = cursor.fetchall()
        conn.close()

        return workflows

    def get_workflows_by_image_type(self, image_type):
        """Recupera workflow filtrati per tipo di immagine"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''
                SELECT * FROM workflows 
                WHERE image_features LIKE ? 
                ORDER BY timestamp DESC
            ''', (f'%"image_type": "{image_type}"%',))

        workflows = cursor.fetchall()
        conn.close()

        return workflows

    def get_most_used_actions(self, limit=10):
        """Trova le azioni più utilizzate dall'utente"""
        workflows = self.get_user_workflows()
        action_count = {}

        for workflow in workflows:
            actions = json.loads(workflow[3])  # action_sequence
            for action in actions:
                action_name = action if isinstance(action, str) else action.get('action', 'unknown')
                action_count[action_name] = action_count.get(action_name, 0) + 1

        # Ordina per frequenza
        sorted_actions = sorted(action_count.items(), key=lambda x: x[1], reverse=True)
        return sorted_actions[:limit]

    def get_total_workflows(self):
        """Conta il numero totale di workflow"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM workflows")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def update_success_rating(self, session_id, rating):
        """Aggiorna la valutazione di successo di un workflow"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''
                UPDATE workflows 
                SET success_rating = ? 
                WHERE session_id = ?
            ''', (rating, session_id))

        conn.commit()
        conn.close()

    def get_workflow_patterns(self):
        """Analizza i pattern nei workflow dell'utente"""
        workflows = self.get_user_workflows()
        patterns = {
            'common_sequences': {},
            'preferred_starting_actions': {},
            'average_workflow_length': 0,
            'most_successful_workflows': []
        }

        total_length = 0
        for workflow in workflows:
            actions = json.loads(workflow[3])
            action_names = [a if isinstance(a, str) else a.get('action', 'unknown') for a in actions]

            # Sequenze comuni (primi 3 passi)
            if len(action_names) >= 3:
                sequence = ' → '.join(action_names[:3])
                patterns['common_sequences'][sequence] = patterns['common_sequences'].get(sequence, 0) + 1

            # Azioni di inizio preferite
            if action_names:
                first_action = action_names[0]
                patterns['preferred_starting_actions'][first_action] = patterns['preferred_starting_actions'].get(
                    first_action, 0) + 1

            total_length += len(action_names)

            # Workflow di successo (rating >= 4)
            if workflow[7] >= 4:  # success_rating
                patterns['most_successful_workflows'].append({
                    'actions': action_names,
                    'rating': workflow[7],
                    'image_type': json.loads(workflow[4]).get('image_type', 'unknown')
                })

        if workflows:
            patterns['average_workflow_length'] = total_length / len(workflows)

        return patterns

class WorkflowPredictor:
    """Predice le prossime azioni basandosi sui dati storici"""

    def __init__(self):
        self.model = None
        self.feature_encoder = LabelEncoder()
        self.action_encoder = LabelEncoder()
        self.is_trained = False

    def prepare_training_data(self, workflows):
        """Prepara i dati per l'addestramento del modello"""
        X = []  # Features
        y = []  # Azioni target

        for workflow in workflows:
            try:
                actions = json.loads(workflow[3])  # action_sequence
                features = json.loads(workflow[4])  # image_features

                # Crea features numeriche dalle caratteristiche dell'immagine
                feature_vector = self._extract_numeric_features(features)

                # Per ogni azione nel workflow, predici la successiva
                action_names = [a if isinstance(a, str) else a.get('action', 'unknown') for a in actions]

                for i in range(len(action_names) - 1):
                    # Features: caratteristiche immagine + azioni precedenti
                    current_features = feature_vector + self._encode_previous_actions(action_names[:i + 1])
                    X.append(current_features)
                    y.append(action_names[i + 1])

            except Exception as e:
                print(f"Errore nel processare workflow: {e}")
                continue

        return np.array(X), np.array(y)

    def _extract_numeric_features(self, features):
        """Estrae features numeriche dalle caratteristiche dell'immagine"""
        numeric_features = []

        # Features numeriche dirette
        numeric_keys = ['brightness', 'contrast', 'aspect_ratio', 'resolution',
                        'saturation_level', 'dynamic_range', 'has_faces']

        for key in numeric_keys:
            value = features.get(key, 0)
            if isinstance(value, (int, float)):
                numeric_features.append(float(value))
            else:
                numeric_features.append(0.0)

        # Encoding delle features categoriche
        categorical_mappings = {
            'image_type': {'portrait': 1, 'landscape': 2, 'square': 3, 'panoramic': 4},
            'exposure_level': {'underexposed': 1, 'normal': 2, 'overexposed': 3},
            'color_temperature': {'cool': 1, 'neutral': 2, 'warm': 3},
            'sharpness': {'soft': 1, 'moderate_sharp': 2, 'sharp': 3},
            'noise_level': {'low_noise': 1, 'moderate_noise': 2, 'high_noise': 3}
        }

        for key, mapping in categorical_mappings.items():
            value = features.get(key, 'unknown')
            numeric_features.append(float(mapping.get(value, 0)))

        return numeric_features

    def _encode_previous_actions(self, previous_actions, max_actions=5):
        """Codifica le azioni precedenti come features numeriche"""
        # Mappa delle azioni comuni
        action_map = {
            'exposure': 1, 'contrast': 2, 'white_balance': 3, 'vibrance': 4,
            'shadows': 5, 'highlights': 6, 'clarity': 7, 'saturation': 8,
            'noise_reduction': 9, 'sharpening': 10, 'vignette': 11
        }

        # Crea vettore delle azioni precedenti (padding con 0)
        action_vector = [0] * max_actions
        for i, action in enumerate(previous_actions[-max_actions:]):
            action_vector[i] = action_map.get(action, 0)

        return action_vector

    def train_model(self, workflows=None):
        """Addestra il modello di machine learning"""
        if workflows is None:
            # Recupera workflow dal tracker
            tracker = WorkflowTracker()
            workflows = tracker.get_user_workflows()

        if len(workflows) < 10:
            print("Non ci sono abbastanza dati per addestrare il modello")
            return False

        print(f"Addestrando modello con {len(workflows)} workflow...")

        X, y = self.prepare_training_data(workflows)

        if len(X) == 0:
            print("Nessun dato valido per l'addestramento")
            return False

        # Addestra il modello
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True

        print(f"Modello addestrato con {len(X)} esempi")
        return True

    def predict_next_actions(self, image_features, current_actions=[], top_k=5):
        """Predice le prossime azioni più probabili"""
        if not self.is_trained or self.model is None:
            return self._fallback_predictions(image_features, current_actions)

        try:
            # Prepara features per la predizione
            feature_vector = self._extract_numeric_features(image_features)
            action_vector = self._encode_previous_actions(current_actions)
            X_pred = np.array([feature_vector + action_vector])

            # Ottieni probabilità per tutte le classi
            probabilities = self.model.predict_proba(X_pred)[0]
            classes = self.model.classes_

            # Ordina per probabilità
            prob_indices = np.argsort(probabilities)[::-1]

            predictions = []
            for i in range(min(top_k, len(prob_indices))):
                idx = prob_indices[i]
                predictions.append({
                    'action': classes[idx],
                    'confidence': probabilities[idx],
                    'reason': 'Basato sui pattern dei tuoi workflow precedenti'
                })

            return predictions

        except Exception as e:
            print(f"Errore nella predizione ML: {e}")
            return self._fallback_predictions(image_features, current_actions)

    def _fallback_predictions(self, image_features, current_actions):
        """Predizioni di fallback basate su regole semplici"""
        predictions = []

        # Logica semplificata basata su regole
        if not current_actions:
            # Prime azioni comuni
            first_actions = ['exposure', 'white_balance', 'contrast']
            for action in first_actions:
                predictions.append({
                    'action': action,
                    'confidence': 0.7,
                    'reason': 'Azione comune di inizio'
                })
        else:
            # Suggerimenti basati sull'ultima azione
            last_action = current_actions[-1] if current_actions else None

            common_followups = {
                'exposure': ['shadows', 'highlights'],
                'white_balance': ['vibrance', 'saturation'],
                'contrast': ['clarity', 'vibrance'],
                'shadows': ['highlights', 'contrast'],
                'highlights': ['shadows', 'contrast']
            }

            if last_action in common_followups:
                for action in common_followups[last_action]:
                    predictions.append({
                        'action': action,
                        'confidence': 0.6,
                        'reason': f'Comunemente segue {last_action}'
                    })

        return predictions[:5]

    def save_model(self, filename):
        """Salva il modello addestrato"""
        if self.model and self.is_trained:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_encoder': self.feature_encoder,
                    'action_encoder': self.action_encoder,
                    'is_trained': self.is_trained
                }, f)
            print(f"Modello salvato in {filename}")

    def load_model(self, filename):
        """Carica un modello precedentemente salvato"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.feature_encoder = data['feature_encoder']
                self.action_encoder = data['action_encoder']
                self.is_trained = data['is_trained']
            print(f"Modello caricato da {filename}")
            return True
        except Exception as e:
            print(f"Errore nel caricamento del modello: {e}")
            return False

class PresetGenerator:
    """Genera preset personalizzati basati sui pattern dell'utente"""

    def __init__(self):
        self.tracker = WorkflowTracker()

    def generate_personal_presets(self, min_usage=3):
        """Genera preset basati sui workflow più utilizzati"""
        patterns = self.tracker.get_workflow_patterns()
        presets = []

        # Preset basati su sequenze comuni
        for sequence, count in patterns['common_sequences'].items():
            if count >= min_usage:
                preset = {
                    'name': f"Il tuo stile - {sequence.replace(' → ', ' + ')}",
                    'description': f"Basato su {count} workflow simili",
                    'actions': sequence.split(' → '),
                    'usage_count': count,
                    'type': 'sequence_based'
                }
                presets.append(preset)

        # Preset basati su workflow di successo
        successful_workflows = patterns['most_successful_workflows']
        if successful_workflows:
            # Raggruppa per tipo di immagine
            by_image_type = {}
            for workflow in successful_workflows:
                img_type = workflow['image_type']
                if img_type not in by_image_type:
                    by_image_type[img_type] = []
                by_image_type[img_type].append(workflow)

            # Crea preset per ogni tipo
            for img_type, workflows in by_image_type.items():
                if len(workflows) >= 2:
                    # Trova azioni comuni
                    common_actions = self._find_common_actions([w['actions'] for w in workflows])

                    preset = {
                        'name': f"Il tuo {img_type.title()} perfetto",
                        'description': f"Basato sui tuoi {len(workflows)} workflow di maggior successo per {img_type}",
                        'actions': common_actions,
                        'image_type': img_type,
                        'success_rate': sum(w['rating'] for w in workflows) / len(workflows),
                        'type': 'success_based'
                    }
                    presets.append(preset)

        return presets

    def _find_common_actions(self, workflow_lists):
        """Trova azioni comuni tra più workflow"""
        if not workflow_lists:
            return []

        # Conta frequenza di ogni azione
        action_count = {}
        total_workflows = len(workflow_lists)

        for workflow in workflow_lists:
            unique_actions = set(workflow)
            for action in unique_actions:
                action_count[action] = action_count.get(action, 0) + 1

        # Restituisci azioni presenti in almeno il 60% dei workflow
        threshold = total_workflows * 0.6
        common_actions = [action for action, count in action_count.items() if count >= threshold]

        # Ordina secondo l'ordine tipico
        action_order = ['exposure', 'white_balance', 'highlights', 'shadows', 'contrast', 'vibrance', 'clarity']
        ordered_common = []

        for action in action_order:
            if action in common_actions:
                ordered_common.append(action)
                common_actions.remove(action)

        # Aggiungi le rimanenti
        ordered_common.extend(common_actions)

        return ordered_common

    def save_presets_to_file(self, presets, filename="personal_presets.json"):
        """Salva i preset in un file JSON"""
        preset_data = {
            'created_date': datetime.now().isoformat(),
            'total_presets': len(presets),
            'presets': presets
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(preset_data, f, indent=2, ensure_ascii=False)

        print(f"Salvati {len(presets)} preset in {filename}")

if __name__ == "__main__":
    """Funzione principale per testare il sistema"""
    ai = WorkflowAI()

    print("=== PHOTOSHOP AI ASSISTANT ===")
    print("Analizza le tue abitudini di editing e suggerisce workflow personalizzati\n")

    while True:
        print("\nOpzioni disponibili:")
        print("1. Analizza una singola immagine")
        print("2. Genera dati di training da un'immagine")
        print("3. Mostra statistiche dei tuoi workflow")
        print("4. Genera preset personalizzati")
        print("5. Addestra modello di predizione")
        print("6. Esci")

        choice = input("\nScegli un'opzione (1-6): ")

        if choice == '1':
            image_path = input("Inserisci il percorso dell'immagine: ")
            if os.path.exists(image_path):
                ai.show_detailed_analysis(image_path)
            else:
                print("File non trovato!")

        elif choice == '2':
            image_path = input("Inserisci il percorso dell'immagine: ")
            if os.path.exists(image_path):
                num_samples = int(input("Numero di workflow da generare (default: 10): ") or "10")
                ai.generate_diverse_samples(image_path, num_samples)
            else:
                print("File non trovato!")

        elif choice == '3':
            patterns = ai.workflow_tracker.get_workflow_patterns()
            print(f"\n=== STATISTICHE DEI TUOI WORKFLOW ===")
            print(f"Workflow totali: {ai.workflow_tracker.get_total_workflows()}")
            print(f"Lunghezza media: {patterns['average_workflow_length']:.1f} azioni")

            print(f"\nAzioni più utilizzate:")
            most_used = ai.workflow_tracker.get_most_used_actions()
            for action, count in most_used[:5]:
                print(f"  • {action}: {count} volte")

            print(f"\nSequenze più comuni:")
            for sequence, count in list(patterns['common_sequences'].items())[:3]:
                print(f"  • {sequence} ({count} volte)")

        elif choice == '4':
            generator = PresetGenerator()
            presets = generator.generate_personal_presets()

            if presets:
                print(f"\n=== PRESET PERSONALIZZATI GENERATI ===")
                for i, preset in enumerate(presets, 1):
                    print(f"\n{i}. {preset['name']}")
                    print(f"   {preset['description']}")
                    print(f"   Azioni: {' → '.join(preset['actions'])}")

                save = input(f"\nSalvare {len(presets)} preset in un file? (s/n): ")
                if save.lower() == 's':
                    generator.save_presets_to_file(presets)
            else:
                print("Non ci sono abbastanza dati per generare preset personalizzati.")

        elif choice == '5':
            ai.train_and_save_model()

        elif choice == '6':
            print("Arrivederci!")
            break

        else:
            print("Opzione non valida!")