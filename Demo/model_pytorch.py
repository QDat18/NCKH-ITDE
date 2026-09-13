import torch
import torch.nn as nn
import timm


class FrequencyArtifactBranch(nn.Module):
    def __init__(self, in_channels=1, out_features=128, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, out_features),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.projection(self.features(x))


class DeepfakeEfficientNet(nn.Module):
    def __init__(
        self,
        model_name='efficientnet_b4',
        pretrained=True,
        frequency_branch=False,
        frequency_channels=1,
        frequency_features=128,
    ):
        super(DeepfakeEfficientNet, self).__init__()
        self.model_name = model_name
        self.frequency_branch = bool(frequency_branch)

        # Load pre-trained EfficientNet
        self.base_model = timm.create_model(model_name, pretrained=pretrained)
        
        # Modify the classification head for binary classification
        # EfficientNet B4 has a larger embedding space
        in_features = self.base_model.get_classifier().in_features
        if self.frequency_branch:
            self.base_model.classifier = nn.Identity()
            self.rgb_projection = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.SiLU(inplace=True),
                nn.Dropout(0.5),
            )
            self.frequency_model = FrequencyArtifactBranch(
                in_channels=frequency_channels,
                out_features=frequency_features,
            )
            self.classifier = nn.Sequential(
                nn.Linear(512 + frequency_features, 256),
                nn.SiLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, 1),
            )
        else:
            self.base_model.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5), # Increased dropout to reduce overfitting/false positives
                nn.Linear(512, 1) # Output 1 value for Binary Cross Entropy with Logits
            )

    def forward(self, x, frequency=None, return_features=False):
        # Extract features using the base model
        features = self.base_model.forward_features(x)
        # Global pooling to get embedding
        pooled_features = self.base_model.forward_head(features, pre_logits=True)

        if self.frequency_branch:
            if frequency is None:
                raise ValueError("frequency tensor is required when frequency_branch=True")
            rgb_features = self.rgb_projection(pooled_features)
            frequency_features = self.frequency_model(frequency)
            fused_features = torch.cat([rgb_features, frequency_features], dim=1)
            logits = self.classifier(fused_features)
            output_features = fused_features
        else:
            # Final classification
            logits = self.base_model.classifier(pooled_features)
            output_features = pooled_features
        
        if return_features:
            return logits, output_features
        return logits

if __name__ == "__main__":
    model = DeepfakeEfficientNet()
    # B4 expects 380x380
    test_input = torch.randn(1, 3, 380, 380)
    output = model(test_input)
    print(f"Model output shape: {output.shape}")

    freq_model = DeepfakeEfficientNet(frequency_branch=True)
    test_freq = torch.randn(1, 1, 380, 380)
    freq_output = freq_model(test_input, test_freq)
    print(f"Frequency model output shape: {freq_output.shape}")
