#pragma once
// libtorch_models.hh
//
//   LibTorchNeuralModels: loads the six TorchScript .pt files produced by
//   `python -m training.train --weights-dir weights/` and plugs them into
//   the NeuralModels interface used by the cotamer simulation.
//
//   Build requires libtorch; see robot_sim/CMakeLists.txt for the optional
//   seqcomm-sim-trained target.

#include "agent_action.hh"
#include <torch/script.h>
#include <string>
#include <vector>
#include <span>
#include <cmath>
#include <stdexcept>

struct LibTorchNeuralModels : NeuralModels {
    int n_agents;
    int action_dim;

    torch::jit::script::Module encoder_m;
    torch::jit::script::Module attn_a_m;
    torch::jit::script::Module attn_w_m;
    torch::jit::script::Module world_model_m;
    torch::jit::script::Module policy_m;
    torch::jit::script::Module critic_m;

    LibTorchNeuralModels(const std::string& weights_dir,
                         int n_agents, int action_dim)
        : n_agents(n_agents), action_dim(action_dim)
    {
        auto load = [&](const char* name) {
            std::string path = weights_dir + "/" + name + ".pt";
            try {
                return torch::jit::load(path);
            } catch (const c10::Error& e) {
                throw std::runtime_error("Failed to load " + path + ": " + e.what());
            }
        };
        encoder_m     = load("encoder");
        attn_a_m      = load("attn_a");
        attn_w_m      = load("attn_w");
        world_model_m = load("world_model");
        policy_m      = load("policy");
        critic_m      = load("critic");

        encoder_m.eval();     attn_a_m.eval();  attn_w_m.eval();
        world_model_m.eval(); policy_m.eval();  critic_m.eval();
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    // span → (1, n) float tensor (clones data so original memory can be freed)
    static torch::Tensor span_to_tensor(std::span<const float> v) {
        return torch::from_blob(
            const_cast<float*>(v.data()),
            {1, static_cast<long>(v.size())},
            torch::kFloat
        ).clone();
    }

    // flat contiguous tensor → std::vector<float>
    static std::vector<float> tensor_to_vec(const torch::Tensor& t) {
        auto flat = t.contiguous().view(-1);
        return {flat.data_ptr<float>(),
                flat.data_ptr<float>() + flat.numel()};
    }

    // ── NeuralModels interface ────────────────────────────────────────────────

    // e(o): raw obs → hidden state h
    std::vector<float> encode(std::span<const float> obs) override {
        torch::NoGradGuard ng;
        auto out = encoder_m.forward({span_to_tensor(obs)})
                       .toTensor().squeeze(0);
        return tensor_to_vec(out);
    }

    // AM_a(h_self, upper_actions): context for policy / critic
    // messages are zero-padded to n_agents so the model sees a fixed-size input
    std::vector<float> attention_a(
        std::span<const float> h_self,
        const std::vector<std::vector<float>>& messages) override
    {
        torch::NoGradGuard ng;
        auto h    = span_to_tensor(h_self);                   // (1, embed)
        auto msgs = torch::zeros({1, n_agents, action_dim});
        for (int i = 0; i < std::min((int)messages.size(), n_agents); ++i) {
            msgs[0][i] = torch::from_blob(
                const_cast<float*>(messages[i].data()),
                {static_cast<long>(messages[i].size())},
                torch::kFloat
            ).clone();
        }
        return tensor_to_vec(
            attn_a_m.forward({h, msgs}).toTensor().squeeze(0));
    }

    // AM_w(enc_obs, actions): context for world model
    std::vector<float> attention_w(
        const std::vector<std::vector<float>>& enc_obs,
        const std::vector<std::vector<float>>& actions) override
    {
        torch::NoGradGuard ng;
        long n  = enc_obs.size();
        long ed = enc_obs[0].size();
        long ad = actions[0].size();

        auto enc_t = torch::zeros({1, n, ed});
        auto act_t = torch::zeros({1, n, ad});
        for (int i = 0; i < n; ++i) {
            enc_t[0][i] = torch::from_blob(
                const_cast<float*>(enc_obs[i].data()), {ed}, torch::kFloat).clone();
            act_t[0][i] = torch::from_blob(
                const_cast<float*>(actions[i].data()), {ad}, torch::kFloat).clone();
        }
        // Matches Python: messages = cat([enc_obs, actions], dim=-1)
        //                 h_self   = enc_obs.mean(dim=1)
        auto msgs   = torch::cat({enc_t, act_t}, /*dim=*/-1);
        auto h_self = enc_t.mean(1);
        return tensor_to_vec(
            attn_w_m.forward({h_self, msgs}).toTensor().squeeze(0));
    }

    // π(·|context): sample action, return (action, log_prob)
    std::pair<std::vector<float>, float>
    policy_sample(std::span<const float> context) override {
        torch::NoGradGuard ng;
        auto ctx    = span_to_tensor(context);
        auto result = policy_m.run_method("sample", ctx);
        auto tup    = result.toTuple();
        auto action = tup->elements()[0].toTensor().squeeze(0);
        float lp    = tup->elements()[1].toTensor().item<float>();
        return {tensor_to_vec(action), lp};
    }

    // log π_old(a | context)
    float policy_log_prob_old(std::span<const float> context,
                              std::span<const float> action) override {
        torch::NoGradGuard ng;
        auto ctx = span_to_tensor(context);
        auto act = span_to_tensor(action);
        return policy_m.run_method("log_prob_of", ctx, act)
                   .toTensor().item<float>();
    }

    // V(context): scalar value estimate
    float critic(std::span<const float> context) override {
        torch::NoGradGuard ng;
        return critic_m.forward({span_to_tensor(context)})
                   .toTensor().item<float>();
    }

    // M(context_w): predict (next_obs_all_flat, reward)
    std::pair<std::vector<float>, float>
    world_model(std::span<const float> context_w) override {
        torch::NoGradGuard ng;
        auto pred = world_model_m.forward({span_to_tensor(context_w)})
                        .toTensor().squeeze(0);           // (N*obs_dim + 1,)
        float reward = pred[-1].item<float>();
        std::vector<float> next_obs(
            pred.data_ptr<float>(),
            pred.data_ptr<float>() + pred.numel() - 1);
        return {next_obs, reward};
    }
};
