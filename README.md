graph TD
    %% Define Styles
    classDef data fill:#2b3137,stroke:#fafbfc,stroke-width:2px,color:#fff;
    classDef model fill:#0366d6,stroke:#fff,stroke-width:2px,color:#fff;
    classDef loss fill:#d73a49,stroke:#fff,stroke-width:2px,color:#fff;
    classDef script fill:#28a745,stroke:#fff,stroke-width:2px,color:#fff;

    %% Data Pipeline
    A[(High-Res Dataset)]:::data -->|data_loader.py| B[Crop & Bicubic Downsample]:::script
    
    %% Tensors
    B -->|Real LR Tensor| C(Generator):::model
    B -->|Real HR Tensor| D(Discriminator):::model
    B -->|Real HR Tensor| E(VGG19 Feature Extractor):::model

    %% Generation
    C -->|Fake HR Tensor| D
    C -->|Fake HR Tensor| E

    %% Loss Calculation (losses.py)
    D -->|Real/Fake Logits| F{Adversarial Loss}:::loss
    E -->|Texture Feature Maps| G{Content Loss}:::loss
    B -.->|Real HR Pixels| H{Pixel L1 Loss}:::loss
    C -.->|Fake HR Pixels| H

    %% Optimization (train.py)
    F --> I[Total Generator Error]:::loss
    G --> I
    H --> I
    
    I -->|Backpropagation| C
    D -->|Classification Error| J[Discriminator Error]:::loss
    J -->|Backpropagation| D
