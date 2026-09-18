// Person B's server. Needs two single-header libs sitting next to this file:
//   httplib.h   -> https://github.com/yhirose/cpp-httplib
//   json.hpp    -> https://github.com/nlohmann/json (single_include/nlohmann/json.hpp)
//
// Build (Linux/Mac, g++):
//   g++ -std=c++17 server.cpp -o server -lpthread
// Build (with OpenSSL if you ever need https, not needed for hackathon):
//   g++ -std=c++17 -DCPPHTTPLIB_OPENSSL_SUPPORT server.cpp -o server -lssl -lcrypto -lpthread
//
// Run:
//   ./server
// Then in another terminal:
//   curl -X POST localhost:8080/predict -H "Content-Type: application/json" \
//        -d '{"model":"heart","features":{"age":55,"chol":230, ...}}'

#include "httplib.h"
#include "json.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using json = nlohmann::json;

struct ModelWeights {
    std::vector<std::string> feature_order;
    std::vector<double> mean;
    std::vector<double> std;
    std::vector<double> weights;
    double bias;
};

std::map<std::string, ModelWeights> g_models; // "heart" -> weights, "stroke" -> weights

ModelWeights load_model(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "FATAL: could not open " << path << std::endl;
        std::exit(1);
    }
    json j;
    f >> j;

    ModelWeights m;
    m.feature_order = j["feature_order"].get<std::vector<std::string>>();
    m.mean = j["mean"].get<std::vector<double>>();
    m.std = j["std"].get<std::vector<double>>();
    m.weights = j["weights"].get<std::vector<double>>();
    m.bias = j["bias"].get<double>();
    return m;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Scores one model given a map of feature_name -> raw value.
// Applies the same standardization used at training time, then
// score = sigmoid(w . x_scaled + b)
double score_model(const ModelWeights& m, const std::map<std::string, double>& features) {
    double z = m.bias;
    for (size_t i = 0; i < m.feature_order.size(); i++) {
        const std::string& name = m.feature_order[i];
        double raw = 0.0;
        auto it = features.find(name);
        if (it != features.end()) raw = it->second;
        // else: missing feature defaults to 0 (pre-scaling) -- fine for a
        // hackathon demo, just make sure the frontend always sends every field.
        double scaled = (raw - m.mean[i]) / m.std[i];
        z += m.weights[i] * scaled;
    }
    return sigmoid(z);
}

std::map<std::string, double> parse_features(const json& j) {
    std::map<std::string, double> out;
    for (auto& [key, val] : j.items()) {
        out[key] = val.get<double>();
    }
    return out;
}

void set_cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

int main() {
    // --- load weights once at startup ---
    g_models["heart"] = load_model("heart_weights.json");
    g_models["stroke"] = load_model("stroke_weights.json");
    std::cout << "Loaded heart + stroke models." << std::endl;

    httplib::Server svr;

    // Preflight for every route (browsers send OPTIONS before POST)
    svr.Options(R"(/.*)", [](const httplib::Request&, httplib::Response& res) {
        set_cors(res);
    });

    // POST /predict  { "features": { "age": 55, "chol": 230, ... } }
    // -> { "heart_risk": 0.42, "stroke_risk": 0.11 }
    svr.Post("/predict", [](const httplib::Request& req, httplib::Response& res) {
        set_cors(res);
        try {
            json body = json::parse(req.body);
            auto features = parse_features(body["features"]);

            json out;
            out["heart_risk"] = score_model(g_models["heart"], features);
            out["stroke_risk"] = score_model(g_models["stroke"], features);
            res.set_content(out.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            json err;
            err["error"] = e.what();
            res.set_content(err.dump(), "application/json");
        }
    });

    // POST /counterfactual
    //   { "features": {...}, "model": "heart", "flip_field": "bp", "flip_value": 120 }
    // -> { "original_risk": 0.42, "new_risk": 0.31, "delta": -0.11 }
    svr.Post("/counterfactual", [](const httplib::Request& req, httplib::Response& res) {
        set_cors(res);
        try {
            json body = json::parse(req.body);
            auto features = parse_features(body["features"]);
            std::string model_name = body["model"].get<std::string>();
            std::string flip_field = body["flip_field"].get<std::string>();
            double flip_value = body["flip_value"].get<double>();

            const ModelWeights& m = g_models.at(model_name);

            double original = score_model(m, features);

            auto flipped = features;
            flipped[flip_field] = flip_value;
            double new_score = score_model(m, flipped);

            json out;
            out["original_risk"] = original;
            out["new_risk"] = new_score;
            out["delta"] = new_score - original;
            res.set_content(out.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            json err;
            err["error"] = e.what();
            res.set_content(err.dump(), "application/json");
        }
    });

    // quick health check for you + Person C while wiring things up
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        set_cors(res);
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    std::cout << "Listening on http://0.0.0.0:8080" << std::endl;
    svr.listen("0.0.0.0", 8080);
}
