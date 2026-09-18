// server.cpp
// Triage backend — loads two MLP models (heart_model.json, stroke_prediction.json)
// exported in this shape:
// {
//   "feature_names": ["age","sex",...],
//   "scaler": { "type": "standard", "mean": [...], "scale": [...] },
//   "layers": [
//     { "activation": "relu",    "W": [[...],[...],...], "b": [...] },
//     { "activation": "relu",    "W": [[...],[...],...], "b": [...] },
//     { "activation": "sigmoid", "W": [[...]],            "b": 0.07 }
//   ]
// }
//
// Endpoints:
//   POST /predict        { "features": { "age":55, "sex":1, ... } }
//                      -> { "heart_risk": 0.xx, "stroke_risk": 0.xx }
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

using json = nlohmann::json;

// ---------- Model representation ----------

struct Layer {
    std::vector<std::vector<double>> W; // shape: [out][in]
    std::vector<double> b;              // shape: [out]  (or single value broadcast if size==1 and out>1... not needed here)
    std::string activation;             // "relu" or "sigmoid"
};

struct MLPModel {
    std::vector<std::string> feature_names;
    std::vector<double> mean;
    std::vector<double> scale;
    std::vector<Layer> layers;
};

static double relu(double x) { return x > 0.0 ? x : 0.0; }
static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Forward pass through the full MLP given a raw (unstandardized) feature vector.
// Standardizes using scaler.mean/scale, then runs each layer: z = W*x + b, then activation.
static double run_mlp(const MLPModel& m, const std::vector<double>& raw_features) {
    // 1. Standardize
    std::vector<double> x(raw_features.size());
    for (size_t i = 0; i < raw_features.size(); ++i) {
        double s = (i < m.scale.size() && m.scale[i] != 0.0) ? m.scale[i] : 1.0;
        double mu = (i < m.mean.size()) ? m.mean[i] : 0.0;
        x[i] = (raw_features[i] - mu) / s;
    }

    // 2. Forward through each layer
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
                next[out_i] = z; // linear fallback
            }
        }
        current = next;
    }

    // Final layer should have exactly 1 output (the risk probability)
    if (current.empty()) return 0.0;
    return current[0];
}

// Load one model file (heart_model.json / stroke_prediction.json format)
static MLPModel load_model(const std::string& path) {
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

    for (const auto& jl : j.at("layers")) {
        Layer layer;
        layer.activation = jl.at("activation").get<std::string>();
        layer.W = jl.at("W").get<std::vector<std::vector<double>>>();

        // b can be a scalar (single-output final layer) or an array
        const auto& jb = jl.at("b");
        if (jb.is_array()) {
            layer.b = jb.get<std::vector<double>>();
        } else {
            layer.b = { jb.get<double>() };
        }
        m.layers.push_back(std::move(layer));
    }

    std::cout << "Loaded model from " << path << " (" << m.feature_names.size()
              << " features, " << m.layers.size() << " layers)" << std::endl;
    return m;
}

// Build the ordered raw feature vector from a JSON "features" object,
// using the model's feature_names order. Missing fields default to 0.
static std::vector<double> extract_features(const MLPModel& m, const json& features_obj) {
    std::vector<double> out;
    out.reserve(m.feature_names.size());
    for (const auto& name : m.feature_names) {
        if (features_obj.contains(name)) {
            out.push_back(features_obj.at(name).get<double>());
        } else {
            out.push_back(0.0);
        }
    }
    return out;
}

int main() {
    MLPModel heart_model = load_model("heart_model.json");
    MLPModel stroke_model = load_model("stroke_prediction.json");
    std::cout << "Loaded heart + stroke models." << std::endl;

    httplib::Server svr;

    // CORS: allow the frontend (any origin) to call this server
    svr.set_default_headers({
        { "Access-Control-Allow-Origin", "*" },
        { "Access-Control-Allow-Headers", "Content-Type" },
        { "Access-Control-Allow-Methods", "POST, GET, OPTIONS" }
    });
    svr.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
    });

    svr.Post("/predict", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            json body = json::parse(req.body);
            json features_obj = body.at("features");

            auto heart_x = extract_features(heart_model, features_obj);
            auto stroke_x = extract_features(stroke_model, features_obj);

            double heart_risk = run_mlp(heart_model, heart_x);
            double stroke_risk = run_mlp(stroke_model, stroke_x);

            json out = {
                { "heart_risk", heart_risk },
                { "stroke_risk", stroke_risk }
            };
            res.set_content(out.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            json err = { { "error", e.what() } };
            res.set_content(err.dump(), "application/json");
        }
    });

    svr.Post("/counterfactual", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            json body = json::parse(req.body);
            json features_obj = body.at("features");
            std::string model_name = body.at("model").get<std::string>();
            std::string flip_field = body.at("flip_field").get<std::string>();
            double flip_value = body.at("flip_value").get<double>();

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
                { "delta", new_risk - original_risk }
            };
            res.set_content(out.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            json err = { { "error", e.what() } };
            res.set_content(err.dump(), "application/json");
        }
    });

    std::cout << "Listening on http://0.0.0.0:8080" << std::endl;
    svr.listen("0.0.0.0", 8080);
    return 0;
}
