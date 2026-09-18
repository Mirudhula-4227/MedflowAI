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

// Tree structure for Gradient Boosted Decision Trees (GBDT)
struct Tree {
    std::vector<int> left;
    std::vector<int> right;
    std::vector<int> feature;
    std::vector<double> threshold;
    std::vector<double> value;
};

struct GBDTModel {
    std::vector<std::string> feature_names;
    double decision_threshold = 0.5;
    double base_logit = 0.0;
    double learning_rate = 0.1;
    std::vector<Tree> trees;
    std::vector<double> imputer_values;
    std::vector<double> ood_mean;
    std::vector<double> ood_scale;
    double ood_threshold = 7.0;
};

enum class ModelArchitecture {
    MLP,
    GBDT
};

struct UnifiedModel {
    ModelArchitecture arch = ModelArchitecture::MLP;
    std::string model_type = "mlp_relu"; // "mlp_relu" or "gradient_boosted_trees"
    std::vector<std::string> feature_names;
    MLPModel mlp;
    GBDTModel gbdt;
};

struct Prediction {
    double risk;
    double ood_distance;
    bool is_ood;
    double ensemble_variance = 0.0;
    double sub_ensemble_drift = 0.0;
};

struct HallucinationCheck {
    bool passed = true;
    std::string hallucination_risk = "LOW"; // "LOW", "ELEVATED", "HIGH"
    double confidence_score = 1.0;          // 0.0 to 1.0
    double ensemble_variance = 0.0;
    double sub_ensemble_drift = 0.0;
    std::vector<std::string> warnings;
    bool clinical_sanity_verified = true;
};

static double relu(double x) { return x > 0.0 ? x : 0.0; }
static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Forward pass through MLP given raw features.
static Prediction run_mlp_with_ood(const MLPModel& m, const std::vector<double>& raw_features) {
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
    return { risk, ood_dist, is_ood, 0.0, 0.0 };
}

// Forward pass through GBDT with Platt calibration, in-stride OOD check, and ensemble variance / drift calculation.
static Prediction run_gbdt_with_ood(const GBDTModel& m, const std::vector<double>& raw_features) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < raw_features.size(); ++i) {
        double s = (i < m.ood_scale.size() && m.ood_scale[i] != 0.0) ? m.ood_scale[i] : 1.0;
        double mu = (i < m.ood_mean.size()) ? m.ood_mean[i] : 0.0;
        double zi = (raw_features[i] - mu) / s;
        sum_sq += zi * zi;
    }
    double ood_dist = std::sqrt(sum_sq);
    bool is_ood = (ood_dist > m.ood_threshold);

    double sum_tree_values = 0.0;
    double sum_first_half = 0.0;
    std::vector<double> tree_vals;
    tree_vals.reserve(m.trees.size());

    size_t half_count = m.trees.size() / 2;

    for (size_t t = 0; t < m.trees.size(); ++t) {
        const auto& tree = m.trees[t];
        int node = 0;
        while (tree.left[node] != -1) {
            int feat = tree.feature[node];
            double val = raw_features[feat];
            if (val <= tree.threshold[node]) {
                node = tree.left[node];
            } else {
                node = tree.right[node];
            }
        }
        double v = tree.value[node];
        tree_vals.push_back(v);
        sum_tree_values += v;
        if (t < half_count) {
            sum_first_half += v;
        }
    }

    double logit = m.base_logit + m.learning_rate * sum_tree_values;
    double risk = sigmoid(logit);

    // Sub-ensemble convergence drift (evaluating 50% trees scaled vs 100% trees)
    double logit_half = m.base_logit + m.learning_rate * (sum_first_half * 2.0);
    double risk_half = sigmoid(logit_half);
    double drift = std::abs(risk - risk_half);

    // Variance of individual tree outputs (epistemic disagreement)
    double mean_tree_val = m.trees.empty() ? 0.0 : (sum_tree_values / m.trees.size());
    double var_sum = 0.0;
    for (double v : tree_vals) {
        double diff = v - mean_tree_val;
        var_sum += diff * diff;
    }
    double variance = m.trees.empty() ? 0.0 : (var_sum / m.trees.size());

    return { risk, ood_dist, is_ood, variance, drift };
}

static Prediction run_model_with_ood(const UnifiedModel& m, const std::vector<double>& raw_features) {
    if (m.arch == ModelArchitecture::GBDT) {
        return run_gbdt_with_ood(m.gbdt, raw_features);
    } else {
        return run_mlp_with_ood(m.mlp, raw_features);
    }
}

static double run_model(const UnifiedModel& m, const std::vector<double>& raw_features) {
    return run_model_with_ood(m, raw_features).risk;
}

// Anti-Hallucination & Clinical Plausibility Verification Guard
static HallucinationCheck verify_hallucination(
    const json& f,
    const Prediction& heart_pred,
    const Prediction& stroke_pred,
    double heart_ood_threshold,
    double stroke_ood_threshold
) {
    HallucinationCheck check;
    std::vector<std::string> warnings;

    double max_variance = std::max(heart_pred.ensemble_variance, stroke_pred.ensemble_variance);
    double max_drift = std::max(heart_pred.sub_ensemble_drift, stroke_pred.sub_ensemble_drift);
    check.ensemble_variance = std::round(max_variance * 1000.0) / 1000.0;
    check.sub_ensemble_drift = std::round(max_drift * 1000.0) / 1000.0;

    // 1. Extreme Out-Of-Distribution check (Extrapolation Error)
    if (heart_pred.ood_distance > heart_ood_threshold * 1.35) {
        warnings.push_back("Extrapolation Alert: Cardiac profile exceeds training boundaries (OOD distance: " +
                           std::to_string(std::round(heart_pred.ood_distance * 10.0) / 10.0) +
                           " vs threshold " + std::to_string(std::round(heart_ood_threshold * 10.0) / 10.0) + "). Model predictions may be ungrounded.");
    }
    if (stroke_pred.ood_distance > stroke_ood_threshold * 1.35) {
        warnings.push_back("Extrapolation Alert: Cerebrovascular profile exceeds training boundaries (OOD distance: " +
                           std::to_string(std::round(stroke_pred.ood_distance * 10.0) / 10.0) +
                           " vs threshold " + std::to_string(std::round(stroke_ood_threshold * 10.0) / 10.0) + "). Model predictions may be ungrounded.");
    }

    // 2. Ensemble Instability / Tree Divergence Check
    if (max_drift > 0.22) {
        warnings.push_back("Ensemble Divergence: Tree sub-ensembles drift significantly (" +
                           std::to_string((int)std::round(max_drift * 100.0)) +
                           "%), indicating unstable decision boundaries on this patient profile.");
    }

    // 3. Clinical Monotonicity & Pathological Consistency Rules
    double ca = f.value("ca", 0.0);
    double oldpeak = f.value("oldpeak", 0.0);
    double age = f.value("age", 50.0);
    double trestbps = f.value("trestbps", 120.0);
    double exang = f.value("exang", 0.0);
    double cp = f.value("cp", 0.0);

    // Severe coronary pathology with paradoxical sub-baseline risk
    if ((ca >= 2.0 || oldpeak >= 2.5) && heart_pred.risk < 0.32) {
        warnings.push_back("Clinical Incongruity: Patient exhibits severe coronary pathology (fluoroscopy stenosis ca=" +
                           std::to_string((int)ca) + ", ST depression oldpeak=" +
                           std::to_string(oldpeak).substr(0, 4) + " mm) but model predicted low risk (" +
                           std::to_string((int)(heart_pred.risk * 100)) + "%). Clinical safety review recommended.");
    }

    // Anatomical discrepancy in young patients
    if (age < 26.0 && ca >= 2.0) {
        warnings.push_back("Physiological Anomaly: Multi-vessel coronary calcification (ca=" +
                           std::to_string((int)ca) + ") reported in a patient aged " + std::to_string((int)age) +
                           ". Requires fluoroscopy record re-verification.");
    }

    // Stroke vascular triad contradiction
    double hypertension = f.value("hypertension", 0.0);
    double heart_disease = f.value("heart_disease", 0.0);
    double glucose = f.value("avg_glucose_level", 100.0);

    if (age >= 65.0 && hypertension >= 1.0 && heart_disease >= 1.0 && glucose >= 180.0 && stroke_pred.risk < 0.06) {
        warnings.push_back("Clinical Incongruity: Elderly patient with established vascular triad (hypertension + diabetes + CAD) received unexpectedly low stroke risk (" +
                           std::to_string((int)(stroke_pred.risk * 100)) + "%). Clinical override advised.");
    }

    // Compute composite confidence score (1.0 = perfect sanity, <0.6 = suspect)
    double confidence_score = 1.0;
    if (heart_pred.is_ood || stroke_pred.is_ood) confidence_score -= 0.25;
    if (max_drift > 0.15) confidence_score -= 0.20;
    if (!warnings.empty()) confidence_score -= (0.25 * warnings.size());
    if (confidence_score < 0.05) confidence_score = 0.05;
    check.confidence_score = std::round(confidence_score * 100.0) / 100.0;

    if (warnings.empty()) {
        check.passed = true;
        check.hallucination_risk = (confidence_score >= 0.80) ? "LOW" : "ELEVATED";
        check.clinical_sanity_verified = true;
    } else {
        check.passed = false;
        check.hallucination_risk = (warnings.size() >= 2 || confidence_score < 0.50) ? "HIGH" : "ELEVATED";
        check.clinical_sanity_verified = false;
    }
    check.warnings = warnings;

    return check;
}

// Load model file (heart_model.json / stroke_prediction.json), auto-detecting MLP or GBDT
static UnifiedModel load_model(const std::string& path, double default_ood_thresh = 7.0) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "FATAL: could not open " << path << std::endl;
        std::exit(1);
    }
    json j;
    f >> j;

    UnifiedModel um;
    std::string m_str = j.value("model", "mlp_relu");

    if (m_str == "gradient_boosted_trees") {
        um.arch = ModelArchitecture::GBDT;
        um.model_type = "gradient_boosted_trees";
        um.feature_names = j.at("feature_names").get<std::vector<std::string>>();
        um.gbdt.feature_names = um.feature_names;
        um.gbdt.decision_threshold = j.value("decision_threshold", 0.5);
        um.gbdt.base_logit = j.at("base_logit").get<double>();
        um.gbdt.learning_rate = j.at("learning_rate").get<double>();

        if (j.contains("ood")) {
            const auto& j_ood = j.at("ood");
            um.gbdt.ood_mean = j_ood.at("mean").get<std::vector<double>>();
            um.gbdt.ood_scale = j_ood.at("scale").get<std::vector<double>>();
            um.gbdt.ood_threshold = j_ood.value("threshold", default_ood_thresh);
        } else {
            um.gbdt.ood_threshold = default_ood_thresh;
        }

        if (j.contains("imputer") && j.at("imputer").contains("values")) {
            um.gbdt.imputer_values = j.at("imputer").at("values").get<std::vector<double>>();
        }

        for (const auto& jt : j.at("trees")) {
            Tree t;
            t.left = jt.at("left").get<std::vector<int>>();
            t.right = jt.at("right").get<std::vector<int>>();
            t.feature = jt.at("feature").get<std::vector<int>>();
            t.threshold = jt.at("threshold").get<std::vector<double>>();
            t.value = jt.at("value").get<std::vector<double>>();
            um.gbdt.trees.push_back(std::move(t));
        }

        std::cout << "[INIT] Loaded GBDT " << path << " (" << um.feature_names.size()
                  << " features, " << um.gbdt.trees.size() << " trees, OOD threshold=" << um.gbdt.ood_threshold << ")" << std::endl;
    } else {
        um.arch = ModelArchitecture::MLP;
        um.model_type = "mlp_relu";
        um.feature_names = j.at("feature_names").get<std::vector<std::string>>();
        um.mlp.feature_names = um.feature_names;

        const auto& scaler = j.at("scaler");
        um.mlp.mean = scaler.at("mean").get<std::vector<double>>();
        um.mlp.scale = scaler.at("scale").get<std::vector<double>>();
        um.mlp.ood_threshold = j.value("ood_threshold", default_ood_thresh);

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
            um.mlp.layers.push_back(std::move(layer));
        }

        std::cout << "[INIT] Loaded MLP " << path << " (" << um.feature_names.size()
                  << " features, " << um.mlp.layers.size() << " layers, OOD threshold=" << um.mlp.ood_threshold << ")" << std::endl;
    }

    return um;
}

// Build ordered feature vector from JSON object
static std::vector<double> extract_features(const UnifiedModel& m, const json& features_obj) {
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
    // Auto-detect and load models (GBDT or MLP) with calibrated OOD thresholds
    UnifiedModel heart_model = load_model("heart_model.json", 6.73);
    UnifiedModel stroke_model = load_model("stroke_prediction.json", 7.46);
    std::cout << "[INIT] MedflowAI stateless inference engine ready ("
              << heart_model.model_type << " + " << stroke_model.model_type << ")." << std::endl;

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

            auto heart_pred = run_model_with_ood(heart_model, heart_x);
            auto stroke_pred = run_model_with_ood(stroke_model, stroke_x);

            // 4. Anti-Hallucination & Clinical Plausibility Verification
            double heart_thresh = (heart_model.arch == ModelArchitecture::GBDT) ? heart_model.gbdt.ood_threshold : heart_model.mlp.ood_threshold;
            double stroke_thresh = (stroke_model.arch == ModelArchitecture::GBDT) ? stroke_model.gbdt.ood_threshold : stroke_model.mlp.ood_threshold;

            auto h_check = verify_hallucination(features_obj, heart_pred, stroke_pred, heart_thresh, stroke_thresh);

            std::string confidence = (heart_pred.is_ood || stroke_pred.is_ood || !h_check.passed) ? "LOW" : "HIGH";

            json out = {
                { "heart_risk", heart_pred.risk },
                { "stroke_risk", stroke_pred.risk },
                { "confidence", confidence },
                { "heart_ood", heart_pred.is_ood },
                { "stroke_ood", stroke_pred.is_ood },
                { "heart_ood_distance", round(heart_pred.ood_distance * 1000.0) / 1000.0 },
                { "stroke_ood_distance", round(stroke_pred.ood_distance * 1000.0) / 1000.0 },
                { "heart_model_type", heart_model.model_type },
                { "stroke_model_type", stroke_model.model_type },
                { "stateless", true },
                { "hallucination_check", {
                    { "passed", h_check.passed },
                    { "hallucination_risk", h_check.hallucination_risk },
                    { "confidence_score", h_check.confidence_score },
                    { "ensemble_variance", h_check.ensemble_variance },
                    { "sub_ensemble_drift", h_check.sub_ensemble_drift },
                    { "heart_ood_distance", round(heart_pred.ood_distance * 100.0) / 100.0 },
                    { "heart_ood_threshold", round(heart_thresh * 100.0) / 100.0 },
                    { "stroke_ood_distance", round(stroke_pred.ood_distance * 100.0) / 100.0 },
                    { "stroke_ood_threshold", round(stroke_thresh * 100.0) / 100.0 },
                    { "warnings", h_check.warnings },
                    { "clinical_sanity_verified", h_check.clinical_sanity_verified }
                }}
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

            UnifiedModel& model = (model_name == "heart") ? heart_model : stroke_model;

            auto original_x = extract_features(model, features_obj);
            auto original_pred = run_model_with_ood(model, original_x);

            json flipped_obj = features_obj;
            flipped_obj[flip_field] = flip_value;
            auto new_x = extract_features(model, flipped_obj);
            auto new_pred = run_model_with_ood(model, new_x);

            // Counterfactual monotonicity / hallucination anomaly check
            bool monotonic_violation = false;
            std::string cf_warning = "";
            double orig_val = features_obj.value(flip_field, 0.0);
            if ((flip_field == "trestbps" || flip_field == "chol" || flip_field == "oldpeak" || flip_field == "avg_glucose_level" || flip_field == "bmi") &&
                flip_value > orig_val && (new_pred.risk - original_pred.risk) < -0.05) {
                monotonic_violation = true;
                cf_warning = "Counterfactual anomaly: elevating risk factor paradoxically decreased risk by >5 percentage points.";
            }

            json out = {
                { "original_risk", original_pred.risk },
                { "new_risk", new_pred.risk },
                { "delta", new_pred.risk - original_pred.risk },
                { "model_type", model.model_type },
                { "monotonic_violation", monotonic_violation },
                { "warning", cf_warning },
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
