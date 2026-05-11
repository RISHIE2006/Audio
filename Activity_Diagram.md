# Audio Verifier - Activity Diagram

This activity diagram illustrates the logical flow of control and decision-making within the application when a user processes an audio file for deepfake detection.

```mermaid
stateDiagram-v2
    %% Theme
    classDef default fill:#F4ECE3,stroke:#A67C52,stroke-width:2px,color:#111827,font-family:Inter;

    [*] --> Idle

    state "User Uploads Audio" as Upload
    state "Generate Local Waveform" as Preview
    state "Send Request to API" as SendAPI
    state "Save Temp File (Multer)" as SaveFile
    state "Spawn Python Process" as Spawn
    state "Extract Acoustic Features" as Extract
    state "Run ML Inference" as Inference
    state "Format Output JSON" as Format
    state "Cleanup Temp File" as Cleanup
    state "Render Threat Assessment" as Render

    Idle --> Upload
    Upload --> Preview
    Preview --> SendAPI: User clicks 'Analyze'
    
    state "Backend Processing" as Backend {
        SendAPI --> SaveFile
        SaveFile --> Spawn
        Spawn --> Extract
        Extract --> Inference
        Inference --> Format
        Format --> Cleanup
    }
    
    Cleanup --> Render: Return JSON to Client
    
    state "Decision: Threat Level" as Threat {
        state "High Probability" as High
        state "Low Probability" as Low
        
        High: Show Red Warning (Synthetic)
        Low: Show Green Badge (Authentic)
    }

    Render --> Threat
    Threat --> [*]
```

### Flow Explanation
* **Initialization:** The app sits in an idle state waiting for user input.
* **Client-Side Preparation:** Upon upload, the UI extracts browser-level audio buffers to display the waveform immediately.
* **Backend Processing:** The analysis is handed off to the backend where it is saved, processed by Python via feature extraction (DSP) and Machine Learning (Inference), and formulated into JSON.
* **Branching & Rendering:** Once the UI receives the results, it branches its UI rendering logic based on the probability score—displaying either a synthetic warning or an authentic success badge.
