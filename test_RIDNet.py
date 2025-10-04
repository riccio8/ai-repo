import torch
from PIL import Image
from torchvision import transforms
from train import RIDNet

# Percorso modello addestrato
model_path = "/content/ridnet_sidd.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica modello
model = RIDNet()  # Assicurati di avere la classe RIDNet importata
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Trasformazioni per l'immagine (come durante training)
transform = transforms.Compose([
    transforms.ToTensor()
])

# Carica una immagine di test
img = Image.open("/content/noisy.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)  # batch dimension

# Passa il modello
with torch.no_grad():
    output = model(img_tensor)

# output è un tensore PyTorch, lo puoi salvare come PNG
output_img = output.squeeze(0).cpu().clamp(0,1)  # rimuove batch, limita valori
output_pil = transforms.ToPILImage()(output_img)
output_pil.save("denoised_test.tiff")
print("Test completato, immagine salvata come denoised_test.png")
