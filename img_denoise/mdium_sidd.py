import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
import numpy as np

# ===============================================================
# 1️⃣ MODIFICHE PER SUPPORTARE RAW IMAGES
# ===============================================================

class RawImageProcessor:
    """Processa immagini RAW in modo semplificato"""
    @staticmethod
    def raw_to_rgb(raw_path):
        # Approccio semplificato per processare RAW
        try:
            # Metodo 1: Prova ad aprire come immagine normale
            img = Image.open(raw_path)
            return img.convert('RGB')
        except:
            # Metodo 2: Usa numpy per processare i dati RAW
            try:
                import rawpy
                with rawpy.imread(raw_path) as raw:
                    rgb = raw.postprocess()
                return Image.fromarray(rgb)
            except:
                # Metodo 3: Fallback - tratta come PNG/JPG
                return Image.open(raw_path).convert('RGB')

# ===============================================================
# 2️⃣ MODIFICA DEL DATASET PER RAW
# ===============================================================

class SIDDRawDataset(Dataset):
    def __init__(self, file_pairs, patch_size=128, augment=True):
        self.files = file_pairs
        self.patch_size = patch_size
        self.augment = augment
        self.to_tensor = T.ToTensor()
        self.raw_processor = RawImageProcessor()

    def random_crop(self, img1, img2):
        # Assicuriamoci che le immagini siano abbastanza grandi
        min_size = min(img1.size[0], img1.size[1], img2.size[0], img2.size[1])
        if min_size < self.patch_size:
            # Resize se troppo piccole
            new_size = (self.patch_size, self.patch_size)
            img1 = T.functional.resize(img1, new_size)
            img2 = T.functional.resize(img2, new_size)
        
        i, j, h, w = T.RandomCrop.get_params(img1, output_size=(self.patch_size, self.patch_size))
        return T.functional.crop(img1, i, j, h, w), T.functional.crop(img2, i, j, h, w)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        noisy_path, gt_path = self.files[idx]
        
        try:
            # Processa immagini RAW
            noisy = self.raw_processor.raw_to_rgb(noisy_path)
            gt = self.raw_processor.raw_to_rgb(gt_path)
            
            # Ritaglio
            noisy, gt = self.random_crop(noisy, gt)

            # Data augmentation
            if self.augment and random.random() > 0.5:
                noisy = T.functional.hflip(noisy)
                gt = T.functional.hflip(gt)
            if self.augment and random.random() > 0.5:
                noisy = T.functional.vflip(noisy)
                gt = T.functional.vflip(gt)

            return self.to_tensor(noisy), self.to_tensor(gt)
            
        except Exception as e:
            print(f"❌ Errore con file {noisy_path}: {e}")
            # Ritorna un fallback
            return torch.randn(3, self.patch_size, self.patch_size), torch.randn(3, self.patch_size, self.patch_size)

# ===============================================================
# 3️⃣ PREPARAZIONE FILE PAIRS PER SIDD MEDIUM RAW
# ===============================================================

def prepare_sidd_medium_pairs(data_dir):
    """
    SIDD Medium Raw ha una struttura diversa:
    SIDD_Medium_Raw/
    ├── GroundTruth/
    │   ├── Scene1/
    │   │   ├: IMG_001.GT_00.RAW
    │   │   └: ...
    │   └── Scene2/
    └── Noisy/
        ├── Scene1/
        │   ├: IMG_001.NOISY_00.RAW
        │   └: ...
        └── Scene2/
    """
    
    pairs = []
    gt_base = os.path.join(data_dir, "GroundTruth")
    noisy_base = os.path.join(data_dir, "Noisy")
    
    if not os.path.exists(gt_base):
        # Prova struttura alternativa
        gt_base = data_dir
        noisy_base = data_dir
    
    # Cerca tutte le scene
    scenes = []
    if os.path.exists(gt_base):
        scenes = [d for d in os.listdir(gt_base) if os.path.isdir(os.path.join(gt_base, d))]
    else:
        # Se non trova la struttura standard, cerca qualsiasi cartella
        scenes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    print(f"🎬 Trovate {len(scenes)} scene")
    
    for scene in scenes:
        scene_gt_path = os.path.join(gt_base, scene) if os.path.exists(gt_base) else os.path.join(data_dir, scene)
        scene_noisy_path = os.path.join(noisy_base, scene) if os.path.exists(noisy_base) else os.path.join(data_dir, scene)
        
        # Cerca file GT
        gt_files = []
        if os.path.exists(scene_gt_path):
            gt_files = [f for f in os.listdir(scene_gt_path) 
                       if any(keyword in f.upper() for keyword in ['GT', 'GROUND', 'CLEAN'])]
        
        # Cerca file Noisy
        noisy_files = []
        if os.path.exists(scene_noisy_path):
            noisy_files = [f for f in os.listdir(scene_noisy_path) 
                          if any(keyword in f.upper() for keyword in ['NOISY', 'NOISE'])]
        
        # Se non trova con keyword, prende tutti i file RAW
        if not gt_files and os.path.exists(scene_gt_path):
            gt_files = [f for f in os.listdir(scene_gt_path) if f.lower().endswith(('.raw', '.dng', '.png', '.jpg', '.jpeg'))]
        
        if not noisy_files and os.path.exists(scene_noisy_path):
            noisy_files = [f for f in os.listdir(scene_noisy_path) if f.lower().endswith(('.raw', '.dng', '.png', '.jpg', '.jpeg'))]
        
        # Crea coppie basate su nomi simili
        for gt_file in gt_files:
            gt_path = os.path.join(scene_gt_path, gt_file)
            
            # Trova il corrispondente noisy
            base_name = gt_file.split('.')[0]  # Es: "IMG_001" da "IMG_001.GT_00.RAW"
            
            matching_noisy = [f for f in noisy_files if base_name in f and 'NOISY' in f.upper()]
            
            if matching_noisy:
                for noisy_file in matching_noisy:
                    noisy_path = os.path.join(scene_noisy_path, noisy_file)
                    pairs.append((noisy_path, gt_path))
            else:
                # Se non trova corrispondenza, prova con qualsiasi file noisy
                if noisy_files:
                    noisy_path = os.path.join(scene_noisy_path, noisy_files[0])
                    pairs.append((noisy_path, gt_path))
    
    return pairs

# ===============================================================
# 4️⃣ CODICE PRINCIPALE MODIFICATO
# ===============================================================

# Usa il percorso del dataset decompresso
DATA_DIR = "SIDD_Medium_Raw"  # Modifica questo percorso

print("🔍 Esplorando la struttura del dataset...")
print("Contenuto della directory:")
for item in os.listdir(DATA_DIR):
    print(f"  📁 {item}")

# Prepara le coppie
print("🔄 Preparando le coppie di immagini...")
pairs = prepare_sidd_medium_pairs(DATA_DIR)

if not pairs:
    print("🤔 Struttura non riconosciuta, proviamo approccio alternativo...")
    # Approccio alternativo: cerca tutti i file e pairali logicamente
    all_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(('.raw', '.dng', '.png', '.jpg', '.jpeg')):
                all_files.append(os.path.join(root, file))
    
    print(f"📄 Trovati {len(all_files)} file totali")
    
    # Separa noisy e gt
    noisy_files = [f for f in all_files if 'NOISY' in f.upper()]
    gt_files = [f for f in all_files if any(kw in f.upper() for kw in ['GT', 'GROUND', 'CLEAN'])]
    
    print(f"📸 File noisy: {len(noisy_files)}")
    print(f"🎯 File ground truth: {len(gt_files)}")
    
    # Crea coppie basate su percorsi simili
    for noisy_file in noisy_files:
        noisy_dir, noisy_name = os.path.split(noisy_file)
        base_name = noisy_name.split('.')[0].replace('NOISY', '').replace('noisy', '')
        
        # Cerca GT corrispondente
        matching_gt = []
        for gt_file in gt_files:
            gt_dir, gt_name = os.path.split(gt_file)
            if base_name in gt_name:
                matching_gt.append(gt_file)
        
        if matching_gt:
            pairs.append((noisy_file, matching_gt[0]))
        elif gt_files:
            # Usa il primo GT disponibile
            pairs.append((noisy_file, gt_files[0]))

print(f"✅ Coppie totali trovate: {len(pairs)}")

if pairs:
    print("🔍 Prime 5 coppie:")
    for i, (noisy, gt) in enumerate(pairs[:5]):
        print(f"  {i+1}. Noisy: {os.path.basename(noisy)}")
        print(f"       GT:   {os.path.basename(gt)}")

# ── Shuffle & split
random.shuffle(pairs)
split = int(0.8 * len(pairs))
train_pairs = pairs[:split]
val_pairs = pairs[split:]

print(f"📊 Divisione dataset:")
print(f"   Training: {len(train_pairs)} coppie")
print(f"   Validation: {len(val_pairs)} coppie")

if len(pairs) == 0:
    raise ValueError("❌ Nessuna coppia trovata! Controlla la struttura del dataset.")

# ── Crea datasets e dataloaders
train_dataset = SIDDRawDataset(train_pairs, patch_size=128, augment=True)
val_dataset = SIDDRawDataset(val_pairs, patch_size=128, augment=False)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)  # Batch più piccolo per RAW
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

# ===============================================================
# 5️⃣ INSTALLAZIONE DIPENDENZE PER RAW
# ===============================================================

def install_raw_dependencies():
    """Installa le librerie necessarie per processare RAW"""
    try:
        import rawpy
        print("✅ rawpy già installato")
    except ImportError:
        print("📦 Installando rawpy...")
        import subprocess
        subprocess.check_call(["pip", "install", "rawpy"])
        print("✅ rawpy installato")

# Installa dipendenze
install_raw_dependencies()

# ===============================================================
# 6️⃣ RESTANTE CODICE (UGUALE)
# ===============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🎯 Usando device: {device}")

model = RIDNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.L1Loss()

# Training loop (uguale al codice originale)
EPOCHS = 80
best_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    # Training
    model.train()
    running_loss = 0.0
    for noisy, gt in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False):
        noisy, gt = noisy.to(device), gt.to(device)
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * noisy.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for noisy, gt in tqdm(val_loader, desc="Validating", leave=False):
            noisy, gt = noisy.to(device), gt.to(device)
            output = model(noisy)
            loss = criterion(output, gt)
            val_loss += loss.item() * noisy.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"📊 Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

    # Save best model
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "ridnet_sidd_medium_raw.pth")
        print(f"💾 Modello salvato! Best val loss: {best_loss:.6f}")

print("🎉 Allenamento completato! Modello salvato come 'ridnet_sidd_medium_raw.pth'")
