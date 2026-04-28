#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include "grad_importance.h"

// ================================================================
// 工具函数
// ================================================================
static bool parse_round_header(const std::string& line, int* r) {
    return sscanf(line.c_str(), "Round %d", r) == 1;
}

static bool parse_data_header(const std::string& line, int* idx) {
    return sscanf(line.c_str(), "Data index %d", idx) == 1;
}

static int parse_value_line(const std::string& line, fixed_t* buf, int buf_start, int buf_max) {
    std::istringstream ss(line);
    std::string token;
    int count = 0;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        if (buf_start + count >= buf_max) break;
        try {
            buf[buf_start + count] = (fixed_t)std::stod(token);
            count++;
        } catch (...) {}
    }
    return count;
}

static bool is_value_line(const std::string& line) {
    if (line.empty()) return false;
    return (line[0] == '-' || line[0] == '+' || (line[0] >= '0' && line[0] <= '9'));
}

static std::string get_data_dir() {
    std::ifstream cfg("data_path.cfg");
    std::string path = "./";
    if (cfg.is_open()) {
        std::getline(cfg, path);
        if (!path.empty() && path.back() == '\r') path.pop_back();
        if (!path.empty() && path.back() != '/' && path.back() != '\\') path += "/";
    }
    return path;
}

struct RoundPos {
    int round_num;
    std::streampos pos;
};

static std::vector<RoundPos> scan_rounds(std::ifstream& f) {
    std::vector<RoundPos> result;
    f.clear();
    f.seekg(0);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        int r;
        if (parse_round_header(line, &r)) {
            result.push_back({ r, f.tellg() });
        }
    }
    return result;
}

static int pick_best_round(const std::vector<RoundPos>& round_list, int target) {
    int best_idx = -1;
    int best_round = -1;
    for (int i = 0; i < (int)round_list.size(); i++) {
        if (round_list[i].round_num == target) return i;
        if (round_list[i].round_num > best_round) {
            best_round = round_list[i].round_num;
            best_idx = i;
        }
    }
    return best_idx;
}

// ================================================================
// 解析本地梯度（修复：加入 real_data_idx 数组保存真实样本 ID）
// ================================================================
static bool load_grad_file(const std::string& filename,
                           fixed_t grads[][GRAD_DIM],
                           int real_data_idx[],
                           int* n_samples,
                           int round_num)
{
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cout << "[ERROR] 无法打开: " << filename << std::endl;
        return false;
    }

    *n_samples = 0;
    std::vector<RoundPos> round_list = scan_rounds(f);
    if (round_list.empty()) return false;

    // 【修复1】严格对齐 Round，不再 -1
    int target = round_num;
    int idx = pick_best_round(round_list, target);
    if (idx < 0) return false;

    f.clear();
    f.seekg(round_list[idx].pos);

    std::string line;
    std::streampos before_initial = f.tellg();
    if (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.find("Initial") == std::string::npos) f.seekg(before_initial);
    }

    int cur_sample = -1;
    int cur_dim = 0;

    while (*n_samples < MAX_SAMPLES && std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        int r, data_idx;
        if (parse_round_header(line, &r)) {
            break;
        }
        else if (parse_data_header(line, &data_idx)) {
            if (cur_sample >= 0) {
                for (int k = cur_dim; k < GRAD_DIM; k++) grads[cur_sample][k] = (fixed_t)0;
                (*n_samples)++;
                if (*n_samples >= MAX_SAMPLES) break;
            }
            cur_sample = *n_samples;
            // 【修复2】保存真实的样本 ID！
            real_data_idx[cur_sample] = data_idx;
            cur_dim = 0;
        }
        else if (is_value_line(line)) {
            if (cur_sample >= 0 && cur_dim < GRAD_DIM) {
                cur_dim += parse_value_line(line, grads[cur_sample], cur_dim, GRAD_DIM);
            }
        }
    }

    if (cur_sample >= 0 && cur_dim > 0 && *n_samples < MAX_SAMPLES) {
        for (int k = cur_dim; k < GRAD_DIM; k++) grads[cur_sample][k] = (fixed_t)0;
        (*n_samples)++;
    }
    return *n_samples > 0;
}

// ================================================================
// 解析全局梯度
// ================================================================
static bool load_global_grad(const std::string& filename, fixed_t global_grad[GRAD_DIM], int round_num) {
    std::ifstream f(filename);
    if (!f.is_open()) return false;

    std::vector<RoundPos> round_list = scan_rounds(f);
    if (round_list.empty()) return false;

    // 全局模型在当前轮次可能还没生成，所以找最接近的一轮（通常是上一轮）
    int target = round_num;
    int idx = pick_best_round(round_list, target);
    if (idx < 0) return false;

    f.clear();
    f.seekg(round_list[idx].pos);
    int total = 0;
    std::string line;

    while (total < GRAD_DIM && std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        int r;
        if (parse_round_header(line, &r)) break;
        if (is_value_line(line)) {
            total += parse_value_line(line, global_grad, total, GRAD_DIM);
        }
    }
    return total > 0;
}

// ================================================================
// 主函数
// ================================================================
int main(int argc, char* argv[]) {
    std::string data_dir = get_data_dir();
    std::string grad_file = data_dir + "grad_client0.txt";
    std::string global_file = data_dir + "global_grad.txt";
    int round_num = 1;
    int buffer_size = TOP_K;
    int speed = MAX_SAMPLES;

    if (argc >= 6) {
        grad_file = std::string(argv[1]);
        global_file = std::string(argv[2]);
        round_num = atoi(argv[3]);
        buffer_size = atoi(argv[4]);
        speed = atoi(argv[5]);
    }

    static fixed_t sample_grads[MAX_SAMPLES][GRAD_DIM];
    static fixed_t global_grad[GRAD_DIM];
    static fixed_t scores[MAX_SAMPLES];
    static int real_data_idx[MAX_SAMPLES]; // 【新增】用于存储真实 ID

    memset(sample_grads, 0, sizeof(sample_grads));
    memset(global_grad, 0, sizeof(global_grad));
    memset(scores, 0, sizeof(scores));
    memset(real_data_idx, 0, sizeof(real_data_idx));

    int n_samples = 0;

    if (!load_grad_file(grad_file, sample_grads, real_data_idx, &n_samples, round_num)) return 1;
    if (!load_global_grad(global_file, global_grad, round_num)) return 1;

    grad_importance(sample_grads, global_grad, scores, n_samples);

    std::vector<std::pair<float, int>> score_idx(n_samples);
    for (int i = 0; i < n_samples; i++) {
        score_idx[i] = { (float)scores[i], i };
    }

    std::sort(score_idx.begin(), score_idx.end(), [](const std::pair<float,int>& a, const std::pair<float,int>& b){
        return a.first > b.first;
    });

    int topk = std::min(buffer_size, n_samples);

    std::string out_path;
    size_t last_slash = grad_file.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        out_path = grad_file.substr(0, last_slash + 1) + "selected_indices.txt";
    } else {
        out_path = "selected_indices.txt";
    }

    std::ofstream sel(out_path);
    if (!sel.is_open()) return 1;

    sel << "Round " << round_num << "\n";
    for (int i = 0; i < topk; i++) {
        int local_idx = score_idx[i].second;
        // 【修复3】直接写出真实的样本 ID，抛弃无用的 base_idx 计算
        int global_idx = real_data_idx[local_idx];
        sel << global_idx << "\n";
    }
    sel.close();

    return 0;
}
