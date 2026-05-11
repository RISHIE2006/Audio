# Audio Verifier - Sequence Diagram

This sequence diagram illustrates the step-by-step chronological interactions between the user, the frontend, the backend, the file system, and the Python machine learning engine during the core **Deepfake Prediction** workflow.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as 🌐 React UI (App.tsx)
    participant API as ⚡ Express Server
    participant FS as 💾 File System
    participant Engine as 🧠 Python Engine

    User->>UI: Uploads Audio File
    activate UI
    UI->>UI: Decodes audio & Generates waveform preview
    User->>UI: Clicks "Analyze"
    
    UI->>API: POST /api/predict (FormData)
    activate API
    
    API->>FS: Saves temporary file via Multer
    activate FS
    FS-->>API: Returns File Path
    deactivate FS
    
    API->>Engine: Spawn process (predict_wrapper.py <path>)
    activate Engine
    
    Engine->>FS: Reads audio file
    activate FS
    FS-->>Engine: Audio Data
    deactivate FS
    
    Engine->>FS: Loads model (deepfake_detector.pkl)
    
    Engine->>Engine: Extracts DSP Features (MFCC, Phase, etc.)
    Engine->>Engine: Runs Random Forest Inference
    Engine-->>API: Prints JSON Results to stdout
    deactivate Engine
    
    API->>FS: Deletes temporary uploaded file
    
    API-->>UI: Returns JSON Result Payload
    deactivate API
    
    UI->>UI: Updates Threat Gauge & Spectral Charts
    UI->>UI: Saves local scan history
    UI-->>User: Displays Final Threat Assessment
    deactivate UI
```

### Flow Explanation
1. The user provides an audio file. The React UI instantly decodes it in the browser to show a waveform.
2. The user initiates the analysis, sending the file to the Node.js Express server.
3. The server temporarily saves the file and invokes a Python child process.
4. Python reads the audio, loads the pre-trained model, calculates acoustic features, and executes inference.
5. The result is passed back to Node.js via `stdout`.
6. Node.js cleans up the filesystem and sends the final JSON payload back to the React client for rendering.
