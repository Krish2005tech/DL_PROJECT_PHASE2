import os
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from models.vim import QuadVisionMamba


def run_inference(image_path, model_weights_path):
    # 1. Setup device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Quad-Vim model on {device}...")

    # Initialize model (same as training)
    model = QuadVisionMamba(num_classes=15)

    # ✅ Check if model file exists
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model file not found: {model_weights_path}")

    # ✅ Load weights properly
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    print("✅ Trained weights loaded successfully!")

    model.to(device)
    model.eval()

    # 2. Prepare the Image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Failed to read image")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    # 3. Run Prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted_class_idx = torch.max(probs, dim=1)

    predicted_class_idx = predicted_class_idx.item()
    confidence = confidence.item()

    print(f"Prediction: Class {predicted_class_idx} | Confidence: {confidence:.4f}")

    # 4. Visualize
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.title(f"Class {predicted_class_idx} | Conf {confidence:.2f}")
    plt.axis('off')

    save_name = "inference_result.png"
    plt.savefig(save_name, bbox_inches='tight')
    print(f"✅ Visual output saved as {save_name}")


if __name__ == "__main__":
    test_image = "datasets/DOTAv1/DOTAv1/images/test/P0006.jpg"
    trained_weights = "model.pth"

    run_inference(test_image, trained_weights)
