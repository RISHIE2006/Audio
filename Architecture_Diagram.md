# Audio Verifier - System Architecture

This diagram breaks down the system into a layered architecture to demonstrate how the technologies stack up from the user interface down to the machine learning engine and storage layers.

```mermaid
flowchart TD
    %% Global Design/Theme Definitions
    classDef frontend fill:#ffffff,stroke:#A67C52,stroke-width:2px,color:#111827,rx:8,ry:8,font-family:Inter;
    classDef backend fill:#111827,stroke:#A67C52,stroke-width:2px,color:#ffffff,rx:8,ry:8,font-family:Inter;
    classDef python fill:#fcfaf8,stroke:#A67C52,stroke-width:2px,stroke-dasharray: 5 5,color:#111827,rx:8,ry:8,font-family:Inter;
    classDef storage fill:#A67C52,stroke:#ffffff,stroke-width:2px,color:#ffffff,rx:4,ry:4,font-weight:bold,font-family:Inter;
    classDef client fill:#F4ECE3,stroke:#A67C52,stroke-width:2px,color:#A67C52,rx:20,ry:20,font-weight:bold,font-family:Inter;

    %% Client Actor
    Client((💻 Web Client <br> Browser)):::client

    %% Presentation Layer
    subgraph PresentationLayer [📱 Presentation Layer - Frontend]
        direction LR
        React[⚛️ React 19 UI]:::frontend
        Tailwind[🎨 Tailwind CSS]:::frontend
        Plotly[📊 Plotly.js Viz]:::frontend
    end

    %% Application Layer
    subgraph ApplicationLayer [⚙️ Application Layer - Node.js Server]
        direction LR
        Express[🚀 Express.js API]:::backend
        Multer[📁 Multer Uploads]:::backend
        Spawn[🔄 Child Process]:::backend
    end

    %% Processing/ML Layer
    subgraph MLEngineLayer [🧠 Machine Learning & DSP Engine - Python]
        direction LR
        Predictor[🔍 Inference Engine <br> predict_wrapper.py]:::python
        Trainer[📈 Training Engine <br> train_model.py]:::python
        Scikit[🛠️ Data Science <br> Libraries]:::python
    end

    %% Data/Storage Layer
    subgraph DataLayer [💾 Data & Storage Layer]
        direction LR
        LocalStore[(Browser <br> LocalStorage)]:::storage
        Model[(🤖 ML Model <br> .pkl)]:::storage
        TempDir[(Temp Dir <br> /uploads)]:::storage
        Datasets[(Audio <br> Datasets)]:::storage
    end

    %% Architectural Interactions
    Client <-->|Interacts| React
    React -.->|Caches State| LocalStore
    
    React <-->|REST API <br> JSON / FormData| Express
    
    Express -->|Routes File| Multer
    Multer -->|Saves| TempDir
    Express -->|Executes Script| Spawn
    
    Spawn -->|Spawns| Predictor
    Spawn -->|Spawns| Trainer
    
    Predictor -.->|Reads Audio| TempDir
    Predictor -.->|Loads Weights| Model
    Predictor -->|Relies on| Scikit
    
    Trainer -.->|Reads Audio| Datasets
    Trainer -.->|Updates Weights| Model
    Trainer -->|Relies on| Scikit
    
    linkStyle default stroke:#A67C52,stroke-width:2px,color:#111827,font-size:12px,font-weight:bold;
```

### Layer Breakdown

1. **Presentation Layer (Frontend):** 
   - A single-page application (SPA) built with **React 19** and bundled with **Vite**. 
   - Styled using **Tailwind CSS**. 
   - Audio waveform and confidence charts are rendered using **Plotly.js**.
   
2. **Application Layer (Backend):**
   - A lightweight **Node.js** server using **Express**. 
   - Middleware like **Multer** handles parsing incoming `multipart/form-data` and storing audio securely.
   - It acts as an API gateway, taking HTTP requests and spawning Python child processes natively through Node's `child_process`.
   
3. **Machine Learning & DSP Engine (Processing):**
   - Isolated **Python** scripts ensure the heavy lifting for Digital Signal Processing (DSP) does not block the Node.js event loop.
   - Consists of two primary modules: `predict_wrapper.py` for live inference and `train_model.py` for retraining. 
   - Relies heavily on underlying data science frameworks (like Scikit-Learn or Librosa).
   
4. **Data & Storage Layer:**
   - **Local Storage:** Browsers cache recent scan histories locally without requiring a remote database.
   - **File System:** Audio datasets for training, temporary user uploads for predicting, and the serialized `deepfake_detector.pkl` model weights are strictly managed on the local file system.
