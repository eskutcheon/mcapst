from style_transfer import StyleTransferAugmentor

# Initialize augmentor
augmentor = StyleTransferAugmentor(
    model_checkpoint="checkpoints/photo_image.pt",
    style_paths=["style1.jpg", "style2.jpg"],
    alpha_c=0.5,
)

# Apply augmentation
stylized_image = augmentor(image_batch)
