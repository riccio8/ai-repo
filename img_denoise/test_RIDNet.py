import torch
from PIL import Image
from torchvision import transforms
from train import RIDNet


model_path = "/content/ridnet_sidd.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica modello
model = RIDNet()  
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

img = Image.open("/content/noisy.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)  

with torch.no_grad():
    output = model(img_tensor)

output_img = output.squeeze(0).cpu().clamp(0,1) 
output_pil = transforms.ToPILImage()(output_img)
output_pil.save("denoised_test.tiff")
print("Test completato, immagine salvata come denoised_test.png")


# Create example inputs for exporting the model. The inputs should be a tuple of tensors.
example_inputs = (torch.randn(1, 3, 128, 128),) # 1 batch, 3 canali non scala di ggrigi e 12*18 (anche altri valori)
onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)

onnx_program.save("image_denoise.onnx")
print("model onnx exported")

import onnx

onnx_model = onnx.load("image_denoise.onnx")
onnx.checker.check_model(onnx_model)
