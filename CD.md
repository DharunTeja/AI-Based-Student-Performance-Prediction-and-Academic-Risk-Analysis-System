classDiagram
    class StreamlitApp {
        +render_home()
        +render_dashboard()
        +render_predict()
        +render_batch()
        +render_history()
    }

    class DataPreprocessor {
        +load_data(filepath: str) DataFrame
        +clean_data(df: DataFrame) DataFrame
        +encode_features(df: DataFrame) DataFrame
        +scale_features(df: DataFrame) DataFrame
        +split_data(df: DataFrame) tuple
    }

    class ModelTrainer {
        -models: dict
        -best_model: object
        +train_models(X_train, y_train)
        +evaluate_models(X_test, y_test) dict
        +save_best_model(path: str)
    }

    class Predictor {
        -model: object
        -scaler: object
        +load_model(path: str)
        +predict_single(student_data: dict) dict
        +predict_batch(df: DataFrame) DataFrame
    }

    class RiskAdvisor {
        +analyze_student(student_data: dict) list
        -check_attendance()
        -check_grades()
        -check_study_habits()
        -check_social_behavior()
    }

    class DatabaseManager {
        -is_supabase_connected: bool
        +save_prediction(data: dict) bool
        +get_history() list
        +clear_history() bool
    }

    %% Relationships
    StreamlitApp --> Predictor : Calls for predictions
    StreamlitApp --> DataPreprocessor : Prepares UI Data
    StreamlitApp --> ModelTrainer : Triggers training
    StreamlitApp --> RiskAdvisor : Fetches recommendations
    StreamlitApp --> DatabaseManager : Reads/Writes data
    Predictor ..> ModelTrainer : Uses generated best_model.pkl
