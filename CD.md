+-------------------------------------------------------------+
|                     PRESENTATION LAYER                       |
|                (Streamlit Frontend / Dashboard)              |
|                                                             |
|   Home • Dashboard • Predict • Batch • History              |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                     APPLICATION LAYER                        |
|                    (Backend Processing)                      |
|                                                             |
|   Data Preprocessing • API Handling • Routing               |
+----------------------+----------------------+---------------+
                       |                      |
                       v                      v
        +--------------------------+   +--------------------------+
        |        ML ENGINE         |   |    RISK ADVISOR AGENT    |
        |                          |   |                          |
        |  • Logistic Regression   |   |  • Rule-based logic      |
        |  • Decision Tree         |   |  • Risk analysis         |
        |  • Random Forest         |   |  • Recommendations       |
        +------------+-------------+   +------------+-------------+
                     |                              |
                     +--------------+---------------+
                                    |
                                    v
+-------------------------------------------------------------+
|                        DATA LAYER                            |
|                                                             |
|   Supabase (PostgreSQL) / Local JSON                        |
|                                                             |
|   Student Records • Predictions • Model Metrics             |
+-------------------------------------------------------------+
