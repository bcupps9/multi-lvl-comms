#pragma once
// libtorch_models.hh
//
//   LibtorchNeuralModels — implements the NeuralModels interface by loading
//   TorchScript modules exported from training/train.py.
//
//   This file is only compiled when USE_TORCH is defined.  Without it the
//   seqcomm-sim binary still builds and runs with RandomNeuralModels.
//
// Build with libtorch:
//   cmake -B build -DUSE_TORCH=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch
//   cmake --build build --target seqcomm-sim
//
// Run with trained weights:
//   ./build/robot_sim/seqcomm-sim --weights ./weights

#ifdef USE_TORCH

#include "agent_action.hh"
#include <torch/script.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace seqcomm {

struct LibtorchNeuralModels : NeuralModels {

    // Construct by loading all six TorchScript modules from weights_dir.
    // embed_dim and action_dim must match what was used during training
    // (read from config.json which train.py writes alongside the .pt files).
    LibtorchNeuralModels(const std::string& weights_dir, int n_agents)
        : n_agents_(n_agents)
    {
        namespace fs = std::filesystem;
        auto load = [&](const std::string& name) {
            fs::path p = fs::path(weights_dir) / name;
            if (!fs::exists(p))
                throw std::runtime_error("weight file not found: " + p.string());
            return torch::jit::load(p.string());
        };

        encoder_     = load("encoder.pt");
        attn_a_      = load("attn_a.pt");
        attn_w_      = load("attn_w.pt");
        world_model_ = load("world_model.pt");
        policy_      = load("policy.pt");    // returns (mean, log_std)
        critic_      = load("critic.pt");

        // Switch all modules to inference mode (disables dropout etc.)
        encoder_.eval();
        attn_a_.eval();
        attn_w_.eval();
        world_model_.eval();
        policy_.eval();
        critic_.eval();

        // Read dims from config.json
        fs::path cfg_path = fs::path(weights_dir) / "config.json";
        if (!fs::exists(cfg_path))
            throw std::runtime_error("config.json not found in " + weights_dir);

        std::ifstream f(cfg_path);
        std::string contents((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
        // Minimal JSON parse: just pull out the integers by key name.
        embed_dim_  = json_int(contents, "embed_dim");
        action_dim_ = json_int(contents, "action_dim");
    }


    // ── NeuralModels interface ─────────────────────────────────────────────

    // e(o): encode raw observation → hidden state h
    std::vector<float> encode(std::span<const float> obs) override {
        torch::NoGradGuard ng;
        auto t = span_to_tensor(obs).unsqueeze(0);              // (1, obs_dim)
        auto h = encoder_.forward({t}).toTensor().squeeze(0);   // (embed_dim,)
        return tensor_to_vec(h);
    }

    // AM_a(h_self, messages) → context for policy / critic
    //
    // messages is a variable-length list of message vectors.  In the
    // negotiation phase they are neighbor hidden states (embed_dim each).
    // In the launching phase they are upper-agent action vectors (action_dim each).
    //
    // We pass them as a (1, n_msgs, msg_dim) tensor to the scripted module.
    std::vector<float> attention_a(
        std::span<const float> h_self,
        const std::vector<std::vector<float>>& messages) override
    {
        torch::NoGradGuard ng;
        auto h_t    = span_to_tensor(h_self).unsqueeze(0);      // (1, embed_dim)
        auto msgs_t = vecs_to_tensor(messages).unsqueeze(0);    // (1, n, msg_dim)
        auto ctx = attn_a_.forward({h_t, msgs_t}).toTensor().squeeze(0);
        return tensor_to_vec(ctx);
    }

    // AM_w(enc_obs_all, actions_all) → context for world model
    //
    // The world-model attention module expects concatenated (enc_obs, action)
    // per agent as messages, with h_self = mean of enc_obs.
    // This mirrors what train_world_model.py does in world_model_loss().
    std::vector<float> attention_w(
        const std::vector<std::vector<float>>& enc_obs,
        const std::vector<std::vector<float>>& actions) override
    {
        torch::NoGradGuard ng;
        int n       = static_cast<int>(enc_obs.size());
        int edim    = static_cast<int>(enc_obs[0].size());
        int adim    = static_cast<int>(actions[0].size());
        int msg_dim = edim + adim;

        // messages: cat(enc_obs[i], actions[i]) per agent  → (1, n, edim+adim)
        auto msgs_t = torch::zeros({1, n, msg_dim});
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < edim; ++j)
                msgs_t[0][i][j] = enc_obs[i][j];
            for (int j = 0; j < adim; ++j)
                msgs_t[0][i][edim + j] = actions[i][j];
        }

        // h_self = mean of enc_obs → (1, edim)
        auto h_t = torch::zeros({1, edim});
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < edim; ++j)
                h_t[0][j] += enc_obs[i][j];
        h_t /= static_cast<float>(n);

        auto ctx = attn_w_.forward({h_t, msgs_t}).toTensor().squeeze(0);
        return tensor_to_vec(ctx);
    }

    // π(·|context): sample action and return (action, log_prob)
    //
    // The scripted policy_ returns (mean, log_std).  We sample via the
    // reparameterisation trick and compute the log_prob analytically so the
    // C++ side never needs the Python Normal distribution class.
    std::pair<std::vector<float>, float>
    policy_sample(std::span<const float> context) override {
        torch::NoGradGuard ng;
        auto ctx_t = span_to_tensor(context).unsqueeze(0);      // (1, embed_dim)

        auto result  = policy_.forward({ctx_t}).toTuple();
        auto mean    = result->elements()[0].toTensor().squeeze(0); // (action_dim,)
        auto log_std = result->elements()[1].toTensor().squeeze(0); // (action_dim,)
        auto std_t   = log_std.exp();

        // action ~ N(mean, std)
        auto noise  = torch::randn_like(mean);
        auto action = mean + std_t * noise;

        // log π(a) = sum_d [ -0.5*(noise_d^2 + log(2π)) - log_std_d ]
        constexpr float LOG_2PI = 1.8378770664f;
        float log_prob = (-0.5f * (noise.pow(2) + LOG_2PI) - log_std)
                             .sum().item<float>();

        return {tensor_to_vec(action), log_prob};
    }

    // log π_old(a|context) — used for the PPO importance weight ratio
    float policy_log_prob_old(
        std::span<const float> context,
        std::span<const float> action) override
    {
        torch::NoGradGuard ng;
        auto ctx_t = span_to_tensor(context).unsqueeze(0);
        auto act_t = span_to_tensor(action);

        auto result  = policy_.forward({ctx_t}).toTuple();
        auto mean    = result->elements()[0].toTensor().squeeze(0);
        auto log_std = result->elements()[1].toTensor().squeeze(0);
        auto std_t   = log_std.exp();

        auto noise = (act_t - mean) / std_t;
        constexpr float LOG_2PI = 1.8378770664f;
        return (-0.5f * (noise.pow(2) + LOG_2PI) - log_std).sum().item<float>();
    }

    // V(context): scalar value estimate
    float critic(std::span<const float> context) override {
        torch::NoGradGuard ng;
        auto ctx_t = span_to_tensor(context).unsqueeze(0);          // (1, embed_dim)
        return critic_.forward({ctx_t}).toTensor().item<float>();
    }

    // M(context_w): world model → (next_obs_all_agents_flat, reward)
    std::pair<std::vector<float>, float>
    world_model(std::span<const float> context) override {
        torch::NoGradGuard ng;
        auto ctx_t = span_to_tensor(context).unsqueeze(0);           // (1, embed_dim)
        auto pred  = world_model_.forward({ctx_t}).toTensor().squeeze(0);
        // last element is the reward prediction; everything before is obs flat
        int n_obs = static_cast<int>(pred.size(0)) - 1;
        std::vector<float> obs_pred(pred.data_ptr<float>(),
                                    pred.data_ptr<float>() + n_obs);
        float reward_pred = pred[n_obs].item<float>();
        return {obs_pred, reward_pred};
    }


    // Reload all modules from disk — call after Python writes updated weights.
    void reload(const std::string& weights_dir) {
        namespace fs = std::filesystem;
        auto load = [&](const std::string& name) {
            fs::path p = fs::path(weights_dir) / name;
            if (!fs::exists(p))
                throw std::runtime_error("weight file not found: " + p.string());
            return torch::jit::load(p.string());
        };
        encoder_     = load("encoder.pt");
        attn_a_      = load("attn_a.pt");
        attn_w_      = load("attn_w.pt");
        world_model_ = load("world_model.pt");
        policy_      = load("policy.pt");
        critic_      = load("critic.pt");
        encoder_.eval(); attn_a_.eval(); attn_w_.eval();
        world_model_.eval(); policy_.eval(); critic_.eval();
    }

private:
    torch::jit::Module encoder_;
    torch::jit::Module attn_a_;
    torch::jit::Module attn_w_;
    torch::jit::Module world_model_;
    torch::jit::Module policy_;
    torch::jit::Module critic_;

    int n_agents_;
    int embed_dim_  = 64;
    int action_dim_ = 1;

    // ── Tensor conversion helpers ──────────────────────────────────────────

    static torch::Tensor span_to_tensor(std::span<const float> v) {
        return torch::from_blob(
            const_cast<float*>(v.data()),
            {static_cast<long>(v.size())},
            torch::kFloat32
        ).clone();   // clone so the tensor owns its memory
    }

    static torch::Tensor vecs_to_tensor(const std::vector<std::vector<float>>& vv) {
        if (vv.empty()) return torch::zeros({0, 0});
        int n    = static_cast<int>(vv.size());
        int dim  = static_cast<int>(vv[0].size());
        auto t   = torch::zeros({n, dim});
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < dim; ++j)
                t[i][j] = vv[i][j];
        return t;
    }

    static std::vector<float> tensor_to_vec(const torch::Tensor& t) {
        auto flat = t.contiguous().flatten();
        return std::vector<float>(flat.data_ptr<float>(),
                                  flat.data_ptr<float>() + flat.numel());
    }

    // Minimal JSON integer extractor — avoids a full JSON library dependency.
    // Looks for  "key": <integer>  and returns the integer.
    static int json_int(const std::string& json, const std::string& key) {
        std::string needle = "\"" + key + "\"";
        auto pos = json.find(needle);
        if (pos == std::string::npos)
            throw std::runtime_error("config.json missing key: " + key);
        pos = json.find(':', pos) + 1;
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
        return std::stoi(json.substr(pos));
    }
};

} // namespace seqcomm

#endif // USE_TORCH
