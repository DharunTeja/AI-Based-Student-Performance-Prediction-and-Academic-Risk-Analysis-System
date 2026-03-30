sequenceDiagram
    actor User as Teacher/Admin
    participant UI as Streamlit UI
    participant P as Predictor (ML)
    participant RA as Risk Advisor
    participant DB as Database Manager

    User->>UI: Enters Student Features (30+ inputs)
    User->>UI: Clicks "Predict Performance"
    
    activate UI
    UI->>P: predict_single(student_data)
    activate P
    note right of P: Loads Random Forest Model & Scaler
    P->>P: Preprocess & Scale features
    P->>P: Run Model Inference
    P-->>UI: Return Result (Pass/Fail) & Probability
    deactivate P

    UI->>RA: analyze_student(student_data)
    activate RA
    note right of RA: Evaluates 10 Academic Rules
    RA-->>UI: Return List of Prioritized Recommendations
    deactivate RA

    UI->>DB: save_prediction(result, probability, timestamp)
    activate DB
    DB-->>UI: Return Success Confirmation
    deactivate DB

    UI-->>User: Render Gauge Chart, Results & Risk Alerts
    deactivate UI
