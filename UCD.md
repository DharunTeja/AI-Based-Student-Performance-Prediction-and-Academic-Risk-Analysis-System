flowchart LR
    %% Actors
    User((Teacher / Admin))
    DB[(Database / Supabase)]

    %% System Boundary
    subgraph System [EduInsight AI System]
        direction TB
        UC1([View Analytics Dashboard])
        UC2([Predict Single Student])
        UC3([Batch Predict via CSV])
        UC4([Train ML Models])
        UC5([View & Export History])
        UC6([Generate Risk Advice])
    end

    %% Actor to Use Case Relationships
    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    User --> UC5

    %% Internal Use Case dependencies
    UC2 -. "Includes" .-> UC6
    
    %% Storage Relationships
    UC2 -- "Saves to" --> DB
    UC3 -- "Saves to" --> DB
    UC5 -- "Reads & Deletes" --> DB

    %% Styling
    classDef system fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef usecase fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    class System system
    class UC1,UC2,UC3,UC4,UC5,UC6 usecase
