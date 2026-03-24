/**
 * API Service Layer — communicates with the Flask backend.
 */
import axios from "axios";

// Use environment variable for API URL in production, or fallback to local dev server
const API_BASE = import.meta.env.VITE_API_URL || "https://eduinsight-ai-7oex.onrender.com/api";

const api = axios.create({
  baseURL: API_BASE,
  headers: { "Content-Type": "application/json" },
});

// ---- System ----
export const getStatus = () => api.get("/status");
export const getStats = () => api.get("/stats");
export const getFeatureDescriptions = () => api.get("/feature-descriptions");

// ---- Dashboard ----
export const getDashboardData = (dataset) => api.get(`/dashboard/${dataset}`);

// ---- Predict ----
export const predictStudent = (data) => api.post("/predict", data);
export const predictBatch = (file) => {
  const formData = new FormData();
  formData.append("file", file);
  return api.post("/predict/batch", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

// ---- Training ----
export const trainModels = () => api.post("/train");

// ---- History ----
export const getHistory = (limit = 100) =>
  api.get("/history", { params: { limit } });
export const clearHistory = () => api.delete("/history/clear");

// ---- Model Metrics ----
export const getModelMetrics = () => api.get("/model-metrics");

export default api;
