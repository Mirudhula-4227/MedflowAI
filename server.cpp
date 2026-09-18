// server.cpp
// MedflowAI Triage Backend — High-Performance & Privacy-Preserving Inference Server
//
// Privacy & Data Minimization Guarantees:
// 1. STATELESS: Zero health data persisted to disk or database. All operations are purely in-memory.
// 2. NO RAW HEALTH LOGGING: Request logging records only method, path, HTTP status, and anonymous session IDs.
// 3. ANONYMOUS: Uses ephemeral session tokens (X-Session-ID); decoupled from any personal identifiers.
// 4. API HARDENING: Thread-safe in-memory rate limiting (60 req/min/IP) and physiological bounds validation.
// 5. OOD / CONFIDENCE: Standardized Euclidean distance to centroid computed in-stride with zero extra memory.
//
// Endpoints:
//   POST /predict        { "features": { "age":55, "sex":1, ... } }
//                      -> { "heart_risk": 0.xx, "stroke_risk": 0.xx, "confidence": "HIGH"|"LOW", ... }
//   POST /counterfactual { "features": {...}, "model": "heart"|"stroke",
//                           "flip_field": "chol", "flip_value": 180 }
//                      -> { "original_risk": 0.xx, "new_risk": 0.xx, "delta": 0.xx }

#include "httplib.h"
#include "json.hpp"
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <deque>
#include <algorithm>

using json = nlohmann::json;

// ---------- Model representation ----------

struct Layer {
    std::vector<std::vector<double>> W; // shape: [out][in]
    std::vector<double> b;              // shape: [out]
    std::string activation;             // "relu" or "sigmoid"
};

struct MLPModel {
    std::vector<std::string> feature_names;
    std::vector<double> mean;
    std::vector<double> scale;
    std::vector<Layer> layers;
    double ood_threshold;               // Distance threshold for OOD flagging
};

struct Prediction {
    double risk;
    double ood_distance;
    bool is_ood;
};

static double relu(double x) { return x > 0.0 ? x : 0.0; }
static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Forward pass through MLP given raw features.
// Standardizes: z = (x - mu) / sigma.
// Also computes standardized Euclidean distance D_std = sqrt(sum(z_i^2)) for OOD check with zero extra allocation.
static Prediction run_mlp_with_ood(const MLPModel& m, const std::vector<double>& raw_features) {
    // 1. Standardize and compute OOD distance
    std::vector<double> x(raw_features.size());
    double sum_sq = 0.0;
    for (size_t i = 0; i < raw_features.size(); ++i) {
        double s = (i < m.scale.size() && m.scale[i] != 0.0) ? m.scale[i] : 1.0;
        double mu = (i < m.mean.size()) ? m.mean[i] : 0.0;
        double zi = (raw_features[i] - mu) / s;
        x[i] = zi;
        sum_sq += zi * zi;
    }
    double ood_dist = std::sqrt(sum_sq);
    bool is_ood = (ood_dist > m.ood_threshold);

    // 2. Forward through layers
    std::vector<double> current = x;
    for (const auto& layer : m.layers) {
        std::vector<double> next(layer.W.size(), 0.0);
        for (size_t out_i = 0; out_i < layer.W.size(); ++out_i) {
            double z = 0.0;
            const auto& row = layer.W[out_i];
            for (size_t in_i = 0; in_i < row.size() && in_i < current.size(); ++in_i) {
                z += row[in_i] * current[in_i];
            }
            double bias = (out_i < layer.b.size()) ? layer.b[out_i] : 0.0;
            z += bias;

            if (layer.activation == "relu") {
                next[out_i] = relu(z);
            } else if (layer.activation == "sigmoid") {
                next[out_i] = sigmoid(z);
            } else {
                next[out_i] = z;
            }
        }
        current = next;
    }

    double risk = current.empty() ? 0.0 : current[0];
    return { risk, ood_dist, is_ood };
}

static double run_mlp(const MLPModel& m, const std::vector<double>& raw_features) {
    return run_mlp_with_ood(m, raw_features).risk;
}

// Load model file (heart_model.json / stroke_prediction.json)
static MLPModel load_model(const std::string& path, double default_ood_thresh = 7.0) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "FATAL: could not open " << path << std::endl;
        std::exit(1);
    }
    json j;
    f >> j;

    MLPModel m;
    m.feature_names = j.at("feature_names").get<std::vector<std::string>>();

    const auto& scaler = j.at("scaler");
    m.mean = scaler.at("mean").get<std::vector<double>>();
    m.scale = scaler.at("scale").get<std::vector<double>>();
    m.ood_threshold = j.value("ood_threshold", default_ood_thresh);

    for (const auto& jl : j.at("layers")) {
        Layer layer;
        layer.activation = jl.at("activation").get<std::string>();
        layer.W = jl.at("W").get<std::vector<std::vector<double>>>();

        const auto& jb = jl.at("b");
        if (jb.is_array()) {
            layer.b = jb.get<std::vector<double>>();
        } else {
            layer.b = { jb.get<double>() };
        }
        m.layers.push_back(std::move(layer));
    }

    std::cout << "[INIT] Loaded " << path << " (" << m.feature_names.size()
              << " features, " << m.layers.size() << " layers, OOD threshold=" << m.ood_threshold << ")" << std::endl;
    return m;
}

// Build ordered feature vector from JSON object
static std::vector<double> extract_features(const MLPModel& m, const json& features_obj) {
    std::vector<double> out;
    out.reserve(m.feature_names.size());
    for (const auto& name : m.feature_names) {
        if (features_obj.contains(name) && features_obj.at(name).is_number()) {
            out.push_back(features_obj.at(name).get<double>());
        } else {
            out.push_back(0.0);
        }
    }
    return out;
}

// ---------- Security & Validation ----------

// Physiological range validation to prevent injection or corrupted inputs
static bool validate_features(const json& f, std::string& err) {
    if (!f.is_object()) {
        err = "'features' must be a valid JSON object.";
        return false;
    }

    auto check_bounded = [&](const std::string& key, double low, double high) {
        if (f.contains(key)) {
            if (!f.at(key).is_number()) {
                err = "Field '" + key + "' must be a valid number.";
                return false;
            }
            double val = f.at(key).get<double>();
            if (!std::isfinite(val)) {
                err = "Field '" + key + "' must be a finite number.";
                return false;
            }
            if (val < low || val > high) {
                err = "Field '" + key + "' is out of physiological range [" +
                      std::to_string((int)low) + ", " + std::to_string((int)high) + "].";
                return false;
            }
        }
        return true;
    };

    if (!check_bounded("age", 1.0, 125.0)) return false;
    if (!check_bounded("trestbps", 50.0, 260.0)) return false;
    if (!check_bounded("chol", 60.0, 650.0)) return false;
    if (!check_bounded("thalach", 40.0, 230.0)) return false;
    if (!check_bounded("oldpeak", 0.0, 10.0)) return false;
    if (!check_bounded("avg_glucose_level", 40.0, 400.0)) return false;
    if (!check_bounded("bmi", 10.0, 75.0)) return false;

    return true;
}

// In-Memory Thread-Safe Sliding Window Rate Limiter
class SimpleRateLimiter {
private:
    std::mutex mtx_;
    std::unordered_map<std::string, std::deque<std::chrono::steady_clock::time_point>> clients_;
    size_t max_requests_;
    std::chrono::seconds window_;

public:
    SimpleRateLimiter(size_t max_reqs = 60, int window_sec = 60)
        : max_requests_(max_reqs), window_(window_sec) {}

    bool allow(const std::string& ip) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto now = std::chrono::steady_clock::now();
        auto& timestamps = clients_[ip];

        while (!timestamps.empty() && (now - timestamps.front()) > window_) {
            timestamps.pop_front();
        }

        if (timestamps.size() >= max_requests_) {
            return false;
        }

        timestamps.push_back(now);
        return true;
    }
};

int main() {
    // Calibrated OOD thresholds from ood_detector.py
    MLPModel heart_model = load_model("heart_model.json", 6.73);
    MLPModel stroke_model = load_model("stroke_prediction.json", 7.46);
    std::cout << "[INIT] MedflowAI stateless inference engine ready." << std::endl;

    httplib::Server svr;
    SimpleRateLimiter rate_limiter(60, 60); // 60 requests per minute per client IP

    // Security and CORS headers
    svr.set_default_headers({
        { "Access-Control-Allow-Origin", "*" },
        { "Access-Control-Allow-Headers", "Content-Type, X-Session-ID" },
        { "Access-Control-Allow-Methods", "POST, GET, OPTIONS" },
        { "Strict-Transport-Security", "max-age=31536000; includeSubDomains" },
        { "X-Content-Type-Options", "nosniff" },
        { "X-Frame-Options", "DENY" }
    });

    // Privacy-Safe Audit Logger (Strictly metadata only — zero raw health values recorded)
    svr.set_logger([](const httplib::Request& req, const httplib::Response& res) {
        std::string sess = req.has_header("X-Session-ID")
                           ? req.get_header_value("X-Session-ID")
                           : "anonymous";
        if (sess.length() > 12) sess = sess.substr(0, 12) + "...";
        std::cout << "[AUDIT] " << req.method << " " << req.path
                  << " | Status: " << res.status
                  << " | Session: " << sess << std::endl;
    });

    svr.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
    });

    svr.Post("/predict", [&](const httplib::Request& req, httplib::Response& res) {
        // 1. Rate limiting check
        std::string client_ip = req.remote_addr.empty() ? "127.0.0.1" : req.remote_addr;
        if (!rate_limiter.allow(client_ip)) {
            res.status = 429;
            json err = { { "error", "Rate limit exceeded (60 requests/minute). Please wait." } };
            res.set_content(err.dump(), "application/json");
            return;
        }

        try {
            json body = json::parse(req.body);
            if (!body.contains("features")) {
                res.status = 400;
                res.set_content(json({{"error", "Missing 'features' in request payload."}}).dump(), "application/json");
                return;
            }

            json features_obj = body.at("features");

            // 2. Input validation & sanitization
            std::string val_err;
            if (!validate_features(features_obj, val_err)) {
                res.status = 400;
                res.set_content(json({{"error", val_err}}).dump(), "application/json");
                return;
            }

            // 3. Forward pass with OOD / confidence estimation
            auto heart_x = extract_features(heart_model, features_obj);
            auto stroke_x = extract_features(stroke_model, features_obj);

            auto heart_pred = run_mlp_with_ood(heart_model, heart_x);
            auto stroke_pred = run_mlp_with_ood(stroke_model, stroke_x);

            std::string confidence = (heart_pred.is_ood || stroke_pred.is_ood) ? "LOW" : "HIGH";

            json out = {
                { "heart_risk", heart_pred.risk },
                { "stroke_risk", stroke_pred.risk },
                { "confidence", confidence },
                { "heart_ood", heart_pred.is_ood },
                { "stroke_ood", stroke_pred.is_ood },
                { "heart_ood_distance", round(heart_pred.ood_distance * 1000.0) / 1000.0 },
                { "stroke_ood_distance", round(stroke_pred.ood_distance * 1000.0) / 1000.0 },
                { "stateless", true }
            };
            res.set_content(out.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            json err = { { "error", e.what() } };
            res.set_content(err.dump(), "application/json");
        }
    });

    svr.Post("/counterfactual", [&](const httplib::Request& req, httplib::Response& res) {
        std::string client_ip = req.remote_addr.empty() ? "127.0.0.1" : req.remote_addr;
        if (!rate_limiter.allow(client_ip)) {
            res.status = 429;
            json err = { { "error", "Rate limit exceeded (60 requests/minute)." } };
            res.set_content(err.dump(), "application/json");
            return;
        }

        try {
            json body = json::parse(req.body);
            json features_obj = body.at("features");
            std::string model_name = body.at("model").get<std::string>();
            std::string flip_field = body.at("flip_field").get<std::string>();
            double flip_value = body.at("flip_value").get<double>();

            std::string val_err;
            if (!validate_features(features_obj, val_err)) {
                res.status = 400;
                res.set_content(json({{"error", val_err}}).dump(), "application/json");
                return;
            }

            MLPModel& model = (model_name == "heart") ? heart_model : stroke_model;

            auto original_x = extract_features(model, features_obj);
            double original_risk = run_mlp(model, original_x);

            json flipped_obj = features_obj;
            flipped_obj[flip_field] = flip_value;
            auto new_x = extract_features(model, flipped_obj);
            double new_risk = run_mlp(model, new_x);

            json out = {
                { "original_risk", original_risk },
                { "new_risk", new_risk },
                { "delta", new_risk - original_risk },
                { "stateless", true }
            };
            res.set_content(out.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            json err = { { "error", e.what() } };
            res.set_content(err.dump(), "application/json");
        }
    });

    std::cout << "[SERVER] Listening on http://0.0.0.0:8080 (Stateless, In-Memory)" << std::endl;
    svr.listen("0.0.0.0", 8080);
    return 0;
}
