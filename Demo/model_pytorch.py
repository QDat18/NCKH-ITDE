import torch
import torch.nn as nn
import timm

class DeepfakeEfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet_b4', pretrained=True):
        super(DeepfakeEfficientNet, self).__init__()
        # Load pre-trained EfficientNet
        self.base_model = timm.create_model(model_name, pretrained=pretrained)
        
        # Modify the classification head for binary classification
        # EfficientNet B4 has a larger embedding space
        in_features = self.base_model.get_classifier().in_features
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # Increased dropout to reduce overfitting/false positives
            nn.Linear(512, 1) # Output 1 value for Binary Cross Entropy with Logits
        )

    def forward(self, x, return_features=False):
        # Extract features using the base model
        features = self.base_model.forward_features(x)
        # Global pooling to get embedding
        pooled_features = self.base_model.forward_head(features, pre_logits=True)
        # Final classification
        logits = self.base_model.classifier(pooled_features)
        
        if return_features:
            return logits, pooled_features
        return logits

if __name__ == "__main__":
    model = DeepfakeEfficientNet()
    # B4 expects 380x380
    test_input = torch.randn(1, 3, 380, 380)
    output = model(test_input)
    print(f"Model output shape: {output.shape}")
