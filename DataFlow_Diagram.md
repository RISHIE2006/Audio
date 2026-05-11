# Audio Verifier - Data Flow Architecture

This visually enhanced diagram illustrates the data flow within the Audio Verifier application, covering both the **Deepfake Analysis** and **Model Retraining** pipelines. 

```mermaid
flowchart TB
    %% Global Design/Theme Definitions matching the application's UI
    classDef userNode fill:#F4ECE3,stroke:#A67C52,stroke-width:2px,color:#A67C52,font-weight:bold,font-family:Inter;
    classDef reactNode fill:#ffffff,stroke:#A67C52,stroke-width:2px,color:#111827,rx:8,ry:8,font-family:Inter;
    classDef serverNode fill:#111827,stroke:#A67C52,stroke-width:2px,color:#ffffff,rx:8,ry:8,font-family:Inter;
    classDef pythonNode fill:#fcfaf8,stroke:#A67C52,stroke-width:2px,stroke-dasharray: 5 5,color:#111827,rx:8,ry:8,font-family:Inter;
    classDef dataNode fill:#A67C52,stroke:#ffffff,stroke-width:2px,color:#ffffff,rx:4,ry:4,font-weight:bold,font-family:Inter;
    
    %% Entities
    U((👤 User)):::userNode

    %% Structural Groupings
    subgraph Frontend [🌐 React Frontend Application]
        direction LR
        UI[💻 UI Interface <br> App.tsx]:::reactNode
        LocalStore[(💾 Local Storage <br> Scan History)]:::dataNode
    end

    subgraph Backend [⚡ Node.js / Express Server]
        direction LR
        API[🔌 API Endpoints <br> server.ts]:::serverNode
        Temp[(📂 Temporary <br> Uploads)]:::dataNode
    end

    subgraph ML [🧠 Python DSP & ML Pipeline]
        direction LR
        Inference[⚙️ Predict Engine <br> predict_wrapper.py]:::pythonNode
        Trainer[🏋️ Training Engine <br> train_model.py]:::pythonNode
        Model[(🤖 ML Model <br> detector.pkl)]:::dataNode
        Data[(📁 Local Audio <br> Datasets)]:::dataNode
    end

    %% Workflow 1: Prediction (Audio Analysis)
    U -- "1️⃣ Upload Audio" --> UI
    UI -- "2️⃣ POST /api/predict" --> API
    API -- "3️⃣ Save file to disk" --> Temp
    API -- "4️⃣ Spawn child process" --> Inference
    Temp -. "Reads file" .-> Inference
    Model -. "Loads weights" .-> Inference
    Inference -- "5️⃣ stdout: JSON Results" --> API
    API -- "6️⃣ Cleanup file" --> Temp
    API -- "7️⃣ Returns Payload" --> UI
    UI -- "8️⃣ Save record" --> LocalStore

    %% Workflow 2: Training (Model Refinement)
    U -- "A️⃣ Trigger Retrain" --> UI
    UI -- "B️⃣ POST /api/train" --> API
    API -- "C️⃣ Spawn child process" --> Trainer
    Data -. "Extracts features" .-> Trainer
    Trainer -- "D️⃣ Overwrites" --> Model
    Trainer -- "E️⃣ stdout: Success Log" --> API
    API -- "F️⃣ Confirms" --> UI

    %% Link aesthetics
    linkStyle default stroke:#A67C52,stroke-width:2px,color:#111827,font-size:12px,font-weight:bold;
```

### Key Upgrades
* **Thematic Colors:** Uses the application's actual color palette (`#A67C52` warm brown accents, `#111827` dark slates, `#F4ECE3` cream backgrounds).
* **Emojis & Icons:** Added visual indicators for better glanceability of nodes.
* **Rounded Corners:** Implemented `rx/ry` curvature on nodes for a modern aesthetic mimicking the app's Tailwind UI corners.
* **Separation of Concerns:** Clear demarcation between data stores (cylinders), standard processing nodes (rectangles), and human actors (circles).
