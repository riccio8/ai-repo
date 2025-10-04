import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

# ===============================================================
# 1️⃣ RIDNet Model Definition
# ===============================================================
class FA_Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        attention = self.sigmoid(residual)
        return x * attention

class RIDNet(nn.Module):
    def __init__(self, in_channels=3, channels=64, num_fa_blocks=4, residual_scale=0.1):
        super().__init__()
        self.residual_scale = residual_scale
        self.entry = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[FA_Block(channels) for _ in range(num_fa_blocks)])
        self.exit = nn.Conv2d(channels, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        features = self.entry(x)
        features = self.blocks(features)
        residual = self.exit(features) * self.residual_scale
        return x - residual

# ===============================================================
# 2️⃣ SIDD Dataset Definition
# ===============================================================
class SIDDataset(Dataset):
    def __init__(self, file_pairs, patch_size=128, augment=True):
        self.files = file_pairs
        self.patch_size = patch_size
        self.augment = augment
        self.to_tensor = T.ToTensor()

        assert len(self.files) > 0, "No file pairs provided!"

    def random_crop(self, img1, img2):
        i, j, h, w = T.RandomCrop.get_params(img1, output_size=(self.patch_size, self.patch_size))
        return T.functional.crop(img1, i, j, h, w), T.functional.crop(img2, i, j, h, w)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        noisy_path, gt_path = self.files[idx]
        noisy = Image.open(noisy_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        noisy, gt = self.random_crop(noisy, gt)

        if self.augment and torch.rand(1) > 0.5:
            noisy = T.functional.hflip(noisy)
            gt = T.functional.hflip(gt)

        return self.to_tensor(noisy), self.to_tensor(gt)

# ===============================================================
# 3️⃣ Prepare file pairs & train/val split
# ===============================================================
DATA_DIR = "/root/.cache/kagglehub/datasets/rajat95gupta/smartphone-image-denoising-dataset/versions/1/SIDD_Small_sRGB_Only/Data"

# ── 3.1 Collect all subdirectories (scenes)
subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
print(f"Found {len(subdirs)} subdirectories (scenes)")
print("Examples:", subdirs[:10])

# ── 3.2 Build file pairs from each subdirectory
pairs = []
for subdir in subdirs:
    subdir_path = os.path.join(DATA_DIR, subdir)
    files = os.listdir(subdir_path)
    
    # Find noisy and ground truth files in this subdirectory
    noisy_files = [f for f in files if "NOISY" in f and f.endswith(".PNG")]
    gt_files = [f for f in files if "GT" in f and f.endswith(".PNG")]
    
    # Create pairs for this scene
    for noisy_file in noisy_files:
        for gt_file in gt_files:
            # Make sure they belong to the same image
            # In SIDD, each scene has one GT and one noisy image
            noisy_path = os.path.join(subdir_path, noisy_file)
            gt_path = os.path.join(subdir_path, gt_file)
            pairs.append((noisy_path, gt_path))

print(f"Total pairs found: {len(pairs)}")

if len(pairs) == 0:
    # Alternative approach: look for specific file patterns
    print("Trying alternative file discovery...")
    for subdir in subdirs:
        subdir_path = os.path.join(DATA_DIR, subdir)
        files = os.listdir(subdir_path)
        
        # Look for any PNG files and pair them logically
        png_files = [f for f in files if f.endswith(".PNG")]
        if len(png_files) == 2:  # Should have exactly 2 files per directory
            # Determine which is noisy and which is GT
            noisy_file = next((f for f in png_files if "NOISY" in f), None)
            gt_file = next((f for f in png_files if "GT" in f), None)
            
            if noisy_file and gt_file:
                pairs.append((os.path.join(subdir_path, noisy_file), 
                             os.path.join(subdir_path, gt_file)))
    
    print(f"Pairs found with alternative method: {len(pairs)}")

# ── 3.3 Show some examples
if pairs:
    print("Example pairs:")
    for i, (noisy, gt) in enumerate(pairs[:5]):
        print(f"  {i+1}. Noisy: {os.path.basename(noisy)}")
        print(f"     GT:   {os.path.basename(gt)}")

# ── 3.4 Shuffle & split
random.shuffle(pairs)
split = int(0.8 * len(pairs))
train_pairs = pairs[:split]
val_pairs = pairs[split:]

print(f"Training pairs: {len(train_pairs)}")
print(f"Validation pairs: {len(val_pairs)}")

if len(pairs) == 0:
    raise ValueError("No file pairs found! Check the dataset structure.")

# ── 3.5 Datasets & loaders
train_dataset = SIDDataset(train_pairs, patch_size=128, augment=True)
val_dataset = SIDDataset(val_pairs, patch_size=128, augment=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

# ===============================================================
# 4️⃣ Model, optimizer & loss
# ===============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = RIDNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.L1Loss()

# ===============================================================
# 5️⃣ Training loop
# ===============================================================
EPOCHS = 80
best_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    # ── 5.1 Training
    model.train()
    running_loss = 0.0
    for noisy, gt in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Training]", leave=False):
        noisy, gt = noisy.to(device), gt.to(device)
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * noisy.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}")

    # ── 5.2 Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for noisy, gt in tqdm(val_loader, desc="Validating", leave=False):
            noisy, gt = noisy.to(device), gt.to(device)
            output = model(noisy)
            loss = criterion(output, gt)
            val_loss += loss.item() * noisy.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Validation Loss = {val_loss:.6f}")

    # ── 5.3 Save best model
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "/content/ridnet_sidd.pth")
        print(f"✅ Model saved! Best val loss: {best_loss:.6f}")

print("🎉 Training complete! Model saved as /content/ridnet_sidd.pth")
