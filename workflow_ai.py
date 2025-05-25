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


class ImageAnalyzer:
    """Analizza le caratteristiche delle immagini"""

    def __init__(self):
        self.label_encoder = LabelEncoder()

    def extract_features(self, image_path):
        """Estrae features base dall'immagine"""
        try:
            # Carica immagine
            image = cv2.imread(image_path)
            if image is None:
                return None

            # Calcola features base
            features = {
                'brightness': np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)),
                'contrast': np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)),
                'aspect_ratio': image.shape[1] / image.shape[0],
                'resolution': image.shape[0] * image.shape[1],
                'dominant_color': self._get_dominant_color(image),
                'has_faces': self._detect_faces(image),
                'image_type': self._classify_image_type(image)
            }

            # Aggiungi metadati EXIF se disponibili
            exif_data = self._extract_exif(image_path)
            features.update(exif_data)

            return features

        except Exception as e:
            print(f"Errore nell'analisi di {image_path}: {e}")
            return None

    def _get_dominant_color(self, image):
        """Trova il colore dominante nell'immagine"""
        data = image.reshape((-1, 3))
        data = np.float32(data)

        # K-means per trovare colori dominanti
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        return centers[0].tolist()

    def _detect_faces(self, image):
        """Rileva se ci sono volti nell'immagine"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces) > 0

    def _classify_image_type(self, image):
        """Classifica il tipo di immagine (ritratto, paesaggio, etc.)"""
        height, width = image.shape[:2]

        if width > height * 1.5:
            return "landscape"
        elif height > width * 1.2:
            return "portrait"
        else:
            return "square"

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


class WorkflowGenerator:
    """Genera workflow realistici con suggerimenti di regolazione dettagliati"""

    def __init__(self):
        # Workflow base per diversi tipi di immagine
        self.workflow_templates = {
            'portrait_with_faces': [
                ['exposure', 'highlights', 'shadows', 'skin_tone', 'eye_enhancement'],
                ['white_balance', 'exposure', 'clarity', 'skin_smoothing', 'eye_brightness'],
                ['contrast', 'exposure', 'white_balance', 'portrait_enhance', 'background_blur'],
                ['exposure', 'vibrance', 'skin_tone', 'teeth_whitening', 'hair_enhance']
            ],
            'portrait_no_faces': [
                ['exposure', 'contrast', 'highlights', 'shadows', 'clarity'],
                ['white_balance', 'exposure', 'vibrance', 'vignette', 'grain'],
                ['contrast', 'brightness', 'saturation', 'sharpening', 'color_grading']
            ],
            'landscape': [
                ['exposure', 'highlights', 'shadows', 'vibrance', 'clarity'],
                ['white_balance', 'contrast', 'saturation', 'graduated_filter', 'orton_effect'],
                ['exposure', 'hdr_tone', 'vibrance', 'landscape_enhance', 'sky_replacement'],
                ['contrast', 'clarity', 'structure', 'color_grading', 'vignette']
            ],
            'square': [
                ['exposure', 'contrast', 'saturation', 'vignette', 'grain'],
                ['white_balance', 'highlights', 'shadows', 'clarity', 'color_pop'],
                ['brightness', 'vibrance', 'sharpening', 'vintage_look', 'border']
            ]
        }

        # Azioni che dipendono dalle condizioni dell'immagine
        self.conditional_actions = {
            'low_brightness': ['brightness', 'exposure', 'shadows'],
            'high_brightness': ['highlights', 'exposure', 'contrast'],
            'low_contrast': ['contrast', 'clarity', 'structure'],
            'high_contrast': ['highlights', 'shadows', 'hdr_tone'],
            'warm_colors': ['white_balance_cool', 'color_temperature'],
            'cool_colors': ['white_balance_warm', 'vibrance']
        }

        self.adjustment_advisor = AdjustmentAdvisor()

    def generate_realistic_workflow(self, image_features):
        """Genera un workflow realistico con valori di regolazione"""
        # Determina il tipo di workflow base
        if image_features['has_faces']:
            base_workflows = self.workflow_templates['portrait_with_faces']
        elif image_features['image_type'] == 'portrait':
            base_workflows = self.workflow_templates['portrait_no_faces']
        elif image_features['image_type'] == 'landscape':
            base_workflows = self.workflow_templates['landscape']
        else:
            base_workflows = self.workflow_templates['square']

        # Seleziona un workflow base casuale
        base_workflow = random.choice(base_workflows).copy()

        # Aggiungi azioni condizionali basate sulle caratteristiche dell'immagine
        additional_actions = self._get_conditional_actions(image_features)

        # Mescola alcune azioni aggiuntive nel workflow
        if additional_actions:
            num_additions = min(2, len(additional_actions))
            for _ in range(num_additions):
                action = random.choice(additional_actions)
                position = random.randint(1, len(base_workflow) - 1)
                base_workflow.insert(position, action)

        # Rimuovi duplicati mantenendo l'ordine
        seen = set()
        filtered_workflow = []
        for action in base_workflow:
            if action not in seen:
                filtered_workflow.append(action)
                seen.add(action)

        # Limita la lunghezza del workflow (3-7 azioni)
        max_length = random.randint(3, 7)
        final_actions = filtered_workflow[:max_length]

        # Aggiungi i parametri di regolazione dettagliati
        detailed_workflow = []
        for action in final_actions:
            adjustment = self.adjustment_advisor.calculate_adjustment(action, image_features)
            detailed_workflow.append({
                'action': adjustment['action'],
                'parameter': adjustment.get('parameter', adjustment['action']),
                'value': adjustment['value'],
                'direction': adjustment['direction'],
                'intensity': adjustment['intensity'],
                'description': self._get_action_description(action, adjustment)
            })

        return detailed_workflow

    def _get_conditional_actions(self, features):
        """Determina azioni aggiuntive basate sulle condizioni dell'immagine"""
        actions = []

        # Basato sulla luminosità
        if features['brightness'] < 80:
            actions.extend(self.conditional_actions['low_brightness'])
        elif features['brightness'] > 180:
            actions.extend(self.conditional_actions['high_brightness'])

        # Basato sul contrasto
        if features['contrast'] < 30:
            actions.extend(self.conditional_actions['low_contrast'])
        elif features['contrast'] > 80:
            actions.extend(self.conditional_actions['high_contrast'])

        # Basato sui colori dominanti
        dominant = features['dominant_color']
        if dominant[0] > dominant[2] + 20:  # Più rosso che blu
            actions.extend(self.conditional_actions['warm_colors'])
        elif dominant[2] > dominant[0] + 20:  # Più blu che rosso
            actions.extend(self.conditional_actions['cool_colors'])

        return actions

    def _get_action_description(self, action, adjustment):
        """Genera una descrizione testuale per l'azione"""
        descriptions = {
            'exposure': f"Regola l'esposizione di {adjustment['value']:.1f} stop ({adjustment['direction']})",
            'contrast': f"Modifica il contrasto del {abs(adjustment['value']):.0f}% ({adjustment['direction']})",
            'white_balance': f"Bilancia il bianco a {adjustment['value']:.0f}K",
            'vibrance': f"Aumenta la vividezza del {abs(adjustment['value']):.0f}%",
            'clarity': f"Aggiungi chiarezza ({adjustment['intensity']} intensity)",
            'shadows': f"Recupera dettagli nelle ombre ({adjustment['intensity']})",
            'highlights': f"Controlla le alte luci ({adjustment['intensity']})"
        }
        return descriptions.get(action, f"Applica regolazione {action}")


class WorkflowTracker:
    """Traccia e memorizza i workflow di editing"""

    def __init__(self, db_path="workflow_data.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Inizializza il database SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workflows (
                id INTEGER PRIMARY KEY,
                image_path TEXT,
                timestamp DATETIME,
                action_sequence TEXT,
                image_features TEXT,
                session_id TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def log_workflow(self, image_path, actions, image_features, session_id):
        """Registra un workflow completato"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO workflows (image_path, timestamp, action_sequence, image_features, session_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            image_path,
            datetime.now(),
            json.dumps(actions),
            json.dumps(image_features),
            session_id
        ))

        conn.commit()
        conn.close()

    def get_workflow_data(self):
        """Recupera tutti i dati di workflow per il training"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM workflows", conn)
        conn.close()
        return df


class WorkflowPredictor:
    """Predice il prossimo step nel workflow"""

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.is_trained = False

        # Workflow comuni predefiniti
        self.common_workflows = {
            'portrait': ['exposure', 'white_balance', 'skin_tone', 'eye_enhancement', 'background_blur'],
            'landscape': ['exposure', 'highlight_shadow', 'vibrance', 'clarity', 'graduated_filter'],
            'macro': ['exposure', 'clarity', 'vibrance', 'sharpening', 'vignette'],
            'street': ['exposure', 'contrast', 'black_white', 'grain', 'vignette']
        }

    def prepare_training_data(self, workflow_df):
        """Prepara i dati per il training"""
        training_data = []

        for _, row in workflow_df.iterrows():
            features = json.loads(row['image_features'])
            actions = json.loads(row['action_sequence'])

            # Crea sequenze di training (stato corrente -> prossima azione)
            for i in range(len(actions) - 1):
                current_state = {
                    'image_type': features.get('image_type', 'unknown'),
                    'brightness': features.get('brightness', 0),
                    'contrast': features.get('contrast', 0),
                    'has_faces': features.get('has_faces', False),
                    'current_step': i,
                    'previous_actions': '_'.join(actions[:i]) if i > 0 else 'start'
                }
                next_action = actions[i + 1]

                training_data.append((current_state, next_action))

        return training_data

    def train(self, workflow_df):
        """Addestra il modello di predizione"""
        if len(workflow_df) == 0:
            print("Nessun dato di training disponibile. Uso workflow predefiniti.")
            return

        training_data = self.prepare_training_data(workflow_df)

        if not training_data:
            return

        # Prepara features e labels
        X = []
        y = []

        for state, action in training_data:
            feature_vector = [
                1 if state['image_type'] == 'portrait' else 0,
                1 if state['image_type'] == 'landscape' else 0,
                state['brightness'] / 255.0,
                state['contrast'] / 100.0,
                1 if state['has_faces'] else 0,
                state['current_step']
            ]
            X.append(feature_vector)
            y.append(action)

        # Addestra il modello
        X = np.array(X)
        y = self.label_encoder.fit_transform(y)

        self.model.fit(X, y)
        self.is_trained = True
        print(f"Modello addestrato su {len(training_data)} esempi")

    def predict_next_actions(self, image_features, current_actions=[]):
        """Predice le prossime 3 azioni più probabili"""

        # Se il modello non è addestrato, usa workflow predefiniti
        if not self.is_trained:
            return self._get_default_workflow(image_features, current_actions)

        # Prepara feature vector
        feature_vector = [
            1 if image_features.get('image_type') == 'portrait' else 0,
            1 if image_features.get('image_type') == 'landscape' else 0,
            image_features.get('brightness', 0) / 255.0,
            image_features.get('contrast', 0) / 100.0,
            1 if image_features.get('has_faces', False) else 0,
            len(current_actions)
        ]

        # Predici probabilità per ogni azione
        probabilities = self.model.predict_proba([feature_vector])[0]

        # Ottieni le top 3 predizioni
        top_indices = np.argsort(probabilities)[-3:][::-1]
        predictions = []

        for idx in top_indices:
            action = self.label_encoder.inverse_transform([idx])[0]
            probability = probabilities[idx]
            predictions.append((action, probability))

        return predictions

    def _get_default_workflow(self, image_features, current_actions):
        """Restituisce workflow predefinito basato sul tipo di immagine"""
        image_type = image_features.get('image_type', 'portrait')

        if image_features.get('has_faces', False):
            workflow = self.common_workflows['portrait']
        elif image_type == 'landscape':
            workflow = self.common_workflows['landscape']
        else:
            workflow = self.common_workflows['portrait']  # default

        # Rimuovi azioni già eseguite
        remaining_actions = [action for action in workflow if action not in current_actions]

        # Restituisci le prossime 3 azioni con probabilità fittizie
        next_actions = remaining_actions[:3]
        return [(action, 0.8 - i * 0.1) for i, action in enumerate(next_actions)]

    def save_model(self, filepath):
        """Salva il modello addestrato"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath):
        """Carica un modello salvato"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.is_trained = model_data['is_trained']
            print("Modello caricato con successo")
        except FileNotFoundError:
            print("File del modello non trovato")


class AdjustmentAdvisor:
    """Calcola i valori ottimali per ogni regolazione basandosi sulle caratteristiche delle immagini"""

    PARAMETER_RANGES = {
        'exposure': (-2.0, 2.0),
        'contrast': (-50, 50),
        'saturation': (-100, 100),
        'highlights': (-100, 100),
        'shadows': (-100, 100),
        'vibrance': (-100, 100),
        'clarity': (-100, 100)
    }

    def calculate_adjustment(self, action, image_features):
        """Calcola la regolazione consigliata per un'azione specifica"""
        recommendation = {
            'action': action,
            'value': 0,
            'direction': 'neutral',
            'intensity': 'medium'
        }

        # Logica di calcolo basato sulle caratteristiche
        if action == 'exposure':
            target_brightness = 125
            current = image_features['brightness']
            delta = (target_brightness - current)/target_brightness
            recommendation['value'] = np.clip(delta * 2, -2, 2)

        elif action == 'contrast':
            target_contrast = 45
            current = image_features['contrast']
            recommendation['value'] = np.clip((target_contrast - current)/2, -50, 50)

        elif action == 'white_balance':
            dominant_color = image_features['dominant_color']
            recommendation['value'] = self._calc_white_balance(dominant_color)

        recommendation['direction'] = 'increase' if recommendation['value'] > 0 else 'decrease'
        recommendation['intensity'] = self._get_intensity(abs(recommendation['value']))

        return recommendation

    def _calc_white_balance(self, dominant_color):
        # Logica per il bilanciamento del bianco
        r, g, b = dominant_color
        temp = 6500 + (b - r) * 100
        return np.clip(temp, 2000, 15000)

    def _get_intensity(self, value):
        if value < 0.3: return 'low'
        if value < 0.7: return 'medium'
        return 'high'

class WorkflowAI:
    """Classe principale che coordina tutti i componenti"""

    def __init__(self):
        self.image_analyzer = ImageAnalyzer()
        self.workflow_tracker = WorkflowTracker()
        self.predictor = WorkflowPredictor()
        self.workflow_generator = WorkflowGenerator()

        # Carica modello esistente se disponibile
        self.predictor.load_model("workflow_model.pkl")

    def analyze_and_predict(self, image_path, current_actions=[]):
        """Analizza un'immagine e predice i prossimi step"""

        # Analizza l'immagine
        print(f"Analizzando {image_path}...")
        features = self.image_analyzer.extract_features(image_path)

        if features is None:
            return None

        # Predici prossime azioni
        predictions = self.predictor.predict_next_actions(features, current_actions)

        return {
            'image_features': features,
            'predictions': predictions
        }

    def generate_sample_workflows(self, image_path, num_samples=5):
        """Genera workflow di esempio per popolare il database con dati variati"""
        features = self.image_analyzer.extract_features(image_path)
        if not features:
            return

        print(f"Generando {num_samples} workflow di esempio per {image_path}")

        for i in range(num_samples):
            # Genera workflow dettagliato
            detailed_workflow = self.workflow_generator.generate_realistic_workflow(features)

            # Estrai solo i nomi delle azioni per il logging
            action_names = [step['action'] for step in detailed_workflow]

            session_id = f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
            self.workflow_tracker.log_workflow(image_path, action_names, features, session_id)
            print(f"  Workflow {i + 1}:")
            for step in detailed_workflow:
                print(f"    - {step['description']}")

    def train_from_data(self):
        """Addestra il modello dai dati esistenti"""
        workflow_data = self.workflow_tracker.get_workflow_data()
        self.predictor.train(workflow_data)

        # Salva il modello addestrato
        self.predictor.save_model("workflow_model.pkl")

    def log_completed_workflow(self, image_path, actions, session_id):
        """Registra un workflow completato per migliorare il modello"""
        features = self.image_analyzer.extract_features(image_path)
        if features:
            self.workflow_tracker.log_workflow(image_path, actions, features, session_id)

    def show_database_stats(self):
        """Mostra statistiche del database"""
        workflow_data = self.workflow_tracker.get_workflow_data()
        if len(workflow_data) == 0:
            print("Nessun dato nel database")
            return

        print(f"\n--- STATISTICHE DATABASE ---")
        print(f"Numero totale di workflow: {len(workflow_data)}")

        # Analizza i tipi di azione più comuni
        all_actions = []
        for _, row in workflow_data.iterrows():
            actions = json.loads(row['action_sequence'])
            all_actions.extend(actions)

        from collections import Counter
        action_counts = Counter(all_actions)
        print(f"\nAzioni più comuni:")
        for action, count in action_counts.most_common(10):
            print(f"  {action}: {count} volte")


# Esempio d'uso migliorato
if __name__ == "__main__":
    # Inizializza il sistema
    ai = WorkflowAI()

    # Percorso dell'immagine di esempio
    image_path = "images/image7.jpg"  # Sostituisci con un percorso reale

    # STEP 1: Genera workflow di esempio (aggiornato per la nuova struttura)
    print("=== GENERAZIONE WORKFLOW DI ESEMPIO ===")
    ai.generate_sample_workflows(image_path, num_samples=5)

    # STEP 2: Addestra il modello con i dati generati
    print("\n=== ADDESTRAMENTO MODELLO ===")
    ai.train_from_data()

    # STEP 3: Mostra statistiche del database (con nuove metriche)
    ai.show_database_stats()

    # STEP 4: Analisi e predizione avanzata
    print(f"\n=== ANALISI DETTAGLIATA E SUGGERIMENTI PER {image_path} ===")
    result = ai.analyze_and_predict(image_path)

    if result:
        features = result['image_features']

        # Visualizza l'analisi tecnica
        print("\n--- ANALISI TECNICA ---")
        print(f"Tipo immagine: {features['image_type'].capitalize()}")
        print(f"Dimensioni: {features['resolution']} px")
        print(f"Luminosità media: {features['brightness']:.1f} (0-255)")
        print(f"Contrasto: {features['contrast']:.1f} (deviazione standard)")
        print(f"Colore dominante (RGB): {[int(x) for x in features['dominant_color']]}")
        print(f"Volti rilevati: {'Sì' if features['has_faces'] else 'No'}")
        print(f"Modello fotocamera: {features.get('camera_model', 'Sconosciuto')}")

        # Visualizza il workflow consigliato con regolazioni
        print("\n--- WORKFLOW CONSIGLIATO CON REGOLAZIONI ---")
        detailed_workflow = ai.workflow_generator.generate_realistic_workflow(features)
        for i, step in enumerate(detailed_workflow, 1):
            print(f"{i}. {step['description']}")
            print(f"   • Parametro: {step['parameter']}")
            print(f"   • Valore: {step['value']:.2f}")
            print(f"   • Direzione: {step['direction'].capitalize()}")
            print(f"   • Intensità: {step['intensity'].capitalize()}")

        # STEP 5: Predizioni del modello e logging
        print("\n--- SUGGERIMENTI DINAMICI DEL MODELLO ---")
        for i, (action, probability) in enumerate(result['predictions'], 1):
            print(f"{i}. {action.ljust(20)} (probabilità: {probability:.1%})")

        # Registra il workflow appropriato (solo nomi azioni per il training)
        action_names = [step['action'] for step in detailed_workflow]
        ai.log_completed_workflow(image_path, action_names, "real_user_session_001")
        print(f"\nWorkflow registrato per il training: {action_names}")

        # Mostra confronto tra generato e predetto
        print("\n--- CONFRONTO WORKFLOW ---")
        print("Generato dal sistema:", [step['action'] for step in detailed_workflow])
        print("Suggerito dal modello:", [action for action, _ in result['predictions']])