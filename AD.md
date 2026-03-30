stateDiagram-v2
    [*] --> StartApp: Launch app.py
    
    state StartApp {
        [*] --> ViewHome
    }
    
    ViewHome --> ChooseAction: Navigate via Sidebar
    
    state ChooseAction {
        [*] --> SinglePrediction: Select "Predict"
        [*] --> BatchPrediction: Select "Batch Upload"
        [*] --> ModelTraining: Select "Model Training"
        [*] --> ViewHistory: Select "History"
        [*] --> ViewDashboard: Select "Dashboard"
    }

    state SinglePrediction {
        InputData: User Inputs 30+ Fields
        PredictResult: ML Infers Risk Level
        GenAdvice: RIsk Advisor Rules Triggered
        InputData --> PredictResult
        PredictResult --> GenAdvice
    }

    state BatchPrediction {
        UploadCSV: User Uploads .CSV
        ProcessRows: Model Predicts for all N rows
        VisualRes: Generate Bulk Charts
        UploadCSV --> ProcessRows
        ProcessRows --> VisualRes
    }

    state ModelTraining {
        LoadData: Load student-mat.csv
        TrainAlgorithms: Train LR, DT, RF
        CompareScores: Evaluate Accuracies
        SaveBest: Export best_model.pkl
        LoadData --> TrainAlgorithms
        TrainAlgorithms --> CompareScores
        CompareScores --> SaveBest
    }

    SinglePrediction --> SaveToDB: Automatically Store
    BatchPrediction --> SaveToDB: Automatically Store

    SaveToDB --> ShowResults: Display to User
    ShowResults --> [*]
    
    ModelTraining --> [*]
    ViewHistory --> [*]
    ViewDashboard --> [*]
