#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using i64 = long long;

constexpr double kInf = std::numeric_limits<double>::infinity();

struct Json {
  enum class Type {
    kNull,
    kNumber,
    kString,
    kArray,
    kObject,
  };

  using Array = std::vector<Json>;
  using Object = std::map<std::string, Json>;

  Type type = Type::kNull;
  double number = 0.0;
  std::string string;
  Array array;
  Object object;

  static Json Null() { return Json(); }

  static Json Number(double value) {
    Json json;
    json.type = Type::kNumber;
    json.number = value;
    return json;
  }

  static Json String(std::string value) {
    Json json;
    json.type = Type::kString;
    json.string = std::move(value);
    return json;
  }

  static Json ArrayValue(Array value) {
    Json json;
    json.type = Type::kArray;
    json.array = std::move(value);
    return json;
  }

  static Json ObjectValue(Object value) {
    Json json;
    json.type = Type::kObject;
    json.object = std::move(value);
    return json;
  }

  bool IsNull() const { return type == Type::kNull; }
  bool IsNumber() const { return type == Type::kNumber; }
  bool IsString() const { return type == Type::kString; }
  bool IsArray() const { return type == Type::kArray; }
  bool IsObject() const { return type == Type::kObject; }

  const Json& At(const std::string& key) const {
    auto it = object.find(key);
    if (it == object.end()) {
      throw std::runtime_error("Missing JSON key: " + key);
    }
    return it->second;
  }
};

class JsonParser {
 public:
  explicit JsonParser(std::string text) : text_(std::move(text)) {}

  Json Parse() {
    SkipWhitespace();
    Json value = ParseValue();
    SkipWhitespace();
    if (pos_ != text_.size()) {
      throw std::runtime_error("Unexpected trailing JSON content");
    }
    return value;
  }

 private:
  Json ParseValue() {
    SkipWhitespace();
    if (pos_ >= text_.size()) {
      throw std::runtime_error("Unexpected end of JSON");
    }
    char c = text_[pos_];
    if (c == 'n') {
      Expect("null");
      return Json::Null();
    }
    if (c == '"') {
      return Json::String(ParseString());
    }
    if (c == '[') {
      return ParseArray();
    }
    if (c == '{') {
      return ParseObject();
    }
    if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
      return Json::Number(ParseNumber());
    }
    throw std::runtime_error("Unsupported JSON token");
  }

  Json ParseArray() {
    Consume('[');
    Json::Array values;
    SkipWhitespace();
    if (TryConsume(']')) {
      return Json::ArrayValue(std::move(values));
    }
    while (true) {
      values.push_back(ParseValue());
      SkipWhitespace();
      if (TryConsume(']')) {
        break;
      }
      Consume(',');
    }
    return Json::ArrayValue(std::move(values));
  }

  Json ParseObject() {
    Consume('{');
    Json::Object values;
    SkipWhitespace();
    if (TryConsume('}')) {
      return Json::ObjectValue(std::move(values));
    }
    while (true) {
      std::string key = ParseString();
      SkipWhitespace();
      Consume(':');
      values.emplace(std::move(key), ParseValue());
      SkipWhitespace();
      if (TryConsume('}')) {
        break;
      }
      Consume(',');
    }
    return Json::ObjectValue(std::move(values));
  }

  double ParseNumber() {
    const std::size_t start = pos_;
    if (text_[pos_] == '-') {
      ++pos_;
    }
    while (pos_ < text_.size() &&
           std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
        ++pos_;
      }
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    return std::stod(text_.substr(start, pos_ - start));
  }

  std::string ParseString() {
    Consume('"');
    std::string out;
    while (pos_ < text_.size()) {
      char c = text_[pos_++];
      if (c == '"') {
        return out;
      }
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (pos_ >= text_.size()) {
        throw std::runtime_error("Invalid JSON escape");
      }
      char escaped = text_[pos_++];
      switch (escaped) {
        case '"':
        case '\\':
        case '/':
          out.push_back(escaped);
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u':
          throw std::runtime_error("Unicode escapes are not supported");
        default:
          throw std::runtime_error("Invalid JSON escape sequence");
      }
    }
    throw std::runtime_error("Unterminated JSON string");
  }

  void SkipWhitespace() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  bool TryConsume(char c) {
    SkipWhitespace();
    if (pos_ < text_.size() && text_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }

  void Consume(char c) {
    SkipWhitespace();
    if (pos_ >= text_.size() || text_[pos_] != c) {
      throw std::runtime_error(std::string("Expected JSON character: ") + c);
    }
    ++pos_;
  }

  void Expect(const char* keyword) {
    while (*keyword != '\0') {
      if (pos_ >= text_.size() || text_[pos_] != *keyword) {
        throw std::runtime_error("Unexpected JSON keyword");
      }
      ++pos_;
      ++keyword;
    }
  }

  std::string text_;
  std::size_t pos_ = 0;
};

struct Tensor {
  i64 width = 0;
  i64 height = 0;
};

struct Op {
  std::string type;
  std::vector<int> inputs;
  std::vector<int> outputs;
  i64 base_cost = 0;
};

struct Problem {
  std::vector<Tensor> tensors;
  std::vector<Op> ops;
  i64 fast_memory_capacity = 0;
  double slow_memory_bandwidth = 1.0;
  i64 native_width = 0;
  i64 native_height = 0;
  std::vector<int> producer;
  std::vector<std::vector<int>> consumers;
};

struct Granularity {
  i64 width = 0;
  i64 height = 0;
  i64 depth = 0;
};

struct Plan {
  std::vector<int> ops;
  Granularity granularity;
  std::optional<std::vector<int>> traversal_order;
  std::vector<int> tensors_to_retain;
  double latency = kInf;

  double Score() const {
    if (ops.empty()) {
      return kInf;
    }
    return latency / static_cast<double>(ops.size());
  }
};

i64 CeilDiv(i64 a, i64 b) { return (a + b - 1) / b; }

bool IsPointwise(const Op& op) { return op.type == "Pointwise"; }
bool IsMatMul(const Op& op) { return op.type == "MatMul"; }

template <typename T>
T JsonInt(const Json& value) {
  if (!value.IsNumber()) {
    throw std::runtime_error("Expected JSON number");
  }
  return static_cast<T>(std::llround(value.number));
}

std::string JsonString(const Json& value) {
  if (!value.IsString()) {
    throw std::runtime_error("Expected JSON string");
  }
  return value.string;
}

std::vector<i64> JsonIntArray(const Json& value) {
  if (!value.IsArray()) {
    throw std::runtime_error("Expected JSON array");
  }
  std::vector<i64> out;
  out.reserve(value.array.size());
  for (const Json& element : value.array) {
    out.push_back(JsonInt<i64>(element));
  }
  return out;
}

std::vector<std::vector<int>> JsonNestedIntArray(const Json& value) {
  if (!value.IsArray()) {
    throw std::runtime_error("Expected JSON array");
  }
  std::vector<std::vector<int>> out;
  out.reserve(value.array.size());
  for (const Json& inner : value.array) {
    if (!inner.IsArray()) {
      throw std::runtime_error("Expected nested JSON array");
    }
    std::vector<int> row;
    row.reserve(inner.array.size());
    for (const Json& element : inner.array) {
      row.push_back(JsonInt<int>(element));
    }
    out.push_back(std::move(row));
  }
  return out;
}

Problem ReadProblem(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Unable to open input file: " + path);
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  Json root = JsonParser(buffer.str()).Parse();
  if (!root.IsObject()) {
    throw std::runtime_error("Problem JSON must be an object");
  }

  const std::vector<i64> widths = JsonIntArray(root.At("widths"));
  const std::vector<i64> heights = JsonIntArray(root.At("heights"));
  const std::vector<std::vector<int>> inputs = JsonNestedIntArray(root.At("inputs"));
  const std::vector<std::vector<int>> outputs = JsonNestedIntArray(root.At("outputs"));
  const std::vector<i64> base_costs = JsonIntArray(root.At("base_costs"));
  const Json& op_types_json = root.At("op_types");
  if (!op_types_json.IsArray()) {
    throw std::runtime_error("op_types must be an array");
  }
  std::vector<std::string> op_types;
  op_types.reserve(op_types_json.array.size());
  for (const Json& entry : op_types_json.array) {
    op_types.push_back(JsonString(entry));
  }

  if (widths.size() != heights.size()) {
    throw std::runtime_error("widths and heights size mismatch");
  }
  if (inputs.size() != outputs.size() || inputs.size() != base_costs.size() ||
      inputs.size() != op_types.size()) {
    throw std::runtime_error("Operation arrays size mismatch");
  }

  Problem problem;
  problem.tensors.resize(widths.size());
  for (std::size_t i = 0; i < widths.size(); ++i) {
    problem.tensors[i] = Tensor{widths[i], heights[i]};
  }
  problem.ops.resize(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    problem.ops[i] = Op{op_types[i], inputs[i], outputs[i], base_costs[i]};
  }

  problem.fast_memory_capacity = JsonInt<i64>(root.At("fast_memory_capacity"));
  problem.slow_memory_bandwidth = root.At("slow_memory_bandwidth").number;
  const std::vector<i64> native = JsonIntArray(root.At("native_granularity"));
  if (native.size() != 2) {
    throw std::runtime_error("native_granularity must have length 2");
  }
  problem.native_width = native[0];
  problem.native_height = native[1];

  problem.producer.assign(problem.tensors.size(), -1);
  problem.consumers.assign(problem.tensors.size(), {});
  for (int op_index = 0; op_index < static_cast<int>(problem.ops.size()); ++op_index) {
    const Op& op = problem.ops[op_index];
    for (int tensor_id : op.outputs) {
      if (tensor_id < 0 || tensor_id >= static_cast<int>(problem.tensors.size())) {
        throw std::runtime_error("Output tensor index out of range");
      }
      if (problem.producer[tensor_id] != -1) {
        throw std::runtime_error("Tensor has multiple producers");
      }
      problem.producer[tensor_id] = op_index;
    }
    for (int tensor_id : op.inputs) {
      if (tensor_id < 0 || tensor_id >= static_cast<int>(problem.tensors.size())) {
        throw std::runtime_error("Input tensor index out of range");
      }
      problem.consumers[tensor_id].push_back(op_index);
    }
  }
  return problem;
}

std::vector<i64> SpatialCandidates(i64 dim, i64 native_dim) {
  std::set<i64, std::greater<i64>> candidates;
  const i64 kMinCandidate = std::min<i64>(16, dim);
  for (i64 value = native_dim; value >= kMinCandidate; value /= 2) {
    candidates.insert(std::min(dim, value));
    if (value <= kMinCandidate) {
      break;
    }
  }
  candidates.insert(std::min(dim, native_dim));
  candidates.insert(kMinCandidate);
  if (dim < native_dim) {
    candidates.insert(dim);
  }
  std::vector<i64> out;
  for (i64 value : candidates) {
    if (value > 0) {
      out.push_back(value);
    }
  }
  return out;
}

std::vector<i64> KCandidates(i64 k_dim, i64 native_dim) {
  std::set<i64, std::greater<i64>> candidates;
  const i64 kMinCandidate = std::min<i64>(16, k_dim);
  candidates.insert(k_dim);
  for (i64 value = kMinCandidate; value < k_dim; value *= 2) {
    candidates.insert(value);
  }
  candidates.insert(std::min(k_dim, native_dim));
  candidates.insert(std::min(k_dim, std::max<i64>(kMinCandidate, native_dim / 2)));
  candidates.insert(kMinCandidate);

  std::vector<i64> out;
  for (i64 value : candidates) {
    if (value > 0 && value <= k_dim) {
      out.push_back(value);
    }
  }
  return out;
}

double BytesToTime(i64 bytes, double bandwidth) {
  return static_cast<double>(bytes) / bandwidth;
}

std::vector<std::pair<int, int>> RasterOrder(i64 rows, i64 cols) {
  std::vector<std::pair<int, int>> order;
  order.reserve(static_cast<std::size_t>(rows * cols));
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      order.emplace_back(r, c);
    }
  }
  return order;
}

std::vector<std::pair<int, int>> SnakeOrder(i64 rows, i64 cols) {
  std::vector<std::pair<int, int>> order;
  order.reserve(static_cast<std::size_t>(rows * cols));
  for (int r = 0; r < rows; ++r) {
    if (r % 2 == 0) {
      for (int c = 0; c < cols; ++c) {
        order.emplace_back(r, c);
      }
    } else {
      for (int c = static_cast<int>(cols) - 1; c >= 0; --c) {
        order.emplace_back(r, c);
      }
    }
  }
  return order;
}

std::vector<int> FlattenTraversalOrder(
    const std::vector<std::pair<int, int>>& order, i64 cols) {
  std::vector<int> flat;
  flat.reserve(order.size());
  for (const auto& [r, c] : order) {
    flat.push_back(static_cast<int>(r * cols + c));
  }
  return flat;
}

bool ChainReadyThrough(const Problem& problem, const std::vector<bool>& scheduled,
                       const std::unordered_set<int>& inside, int op_index) {
  for (int tensor_id : problem.ops[op_index].inputs) {
    int producer = problem.producer[tensor_id];
    if (producer != -1 && !scheduled[producer] && !inside.count(producer)) {
      return false;
    }
  }
  return true;
}

std::vector<int> BuildMaxChain(const Problem& problem,
                               const std::vector<bool>& scheduled,
                               int root) {
  std::vector<int> chain{root};
  std::unordered_set<int> inside{root};

  while (true) {
    const int current = chain.back();
    const Op& op = problem.ops[current];
    if (op.outputs.size() != 1) {
      break;
    }
    int output_tensor = op.outputs[0];
    if (problem.consumers[output_tensor].size() != 1) {
      break;
    }
    int next = problem.consumers[output_tensor][0];
    if (scheduled[next] || inside.count(next)) {
      break;
    }
    if (!IsPointwise(problem.ops[next])) {
      break;
    }
    if (!ChainReadyThrough(problem, scheduled, inside, next)) {
      break;
    }
    chain.push_back(next);
    inside.insert(next);
  }

  return chain;
}

std::vector<int> ExternalInputs(const Problem& problem,
                                const std::vector<int>& ops) {
  std::unordered_set<int> inside(ops.begin(), ops.end());
  std::vector<int> external;
  std::unordered_set<int> seen;
  for (int op_index : ops) {
    for (int tensor_id : problem.ops[op_index].inputs) {
      int producer = problem.producer[tensor_id];
      if (producer == -1 || !inside.count(producer)) {
        if (seen.insert(tensor_id).second) {
          external.push_back(tensor_id);
        }
      }
    }
  }
  return external;
}

std::optional<int> SingleOutputTensor(const Problem& problem,
                                      const std::vector<int>& ops) {
  const Op& last = problem.ops[ops.back()];
  if (last.outputs.size() != 1) {
    return std::nullopt;
  }
  return last.outputs[0];
}

Plan EvaluatePointwiseChain(const Problem& problem, const std::vector<int>& ops,
                            const std::unordered_set<int>& retained_inputs,
                            bool retain_final_output) {
  Plan best;
  best.ops = ops;

  const std::optional<int> output_tensor_id = SingleOutputTensor(problem, ops);
  if (!output_tensor_id.has_value()) {
    return best;
  }
  const Tensor& output = problem.tensors[*output_tensor_id];
  const std::vector<int> external_inputs = ExternalInputs(problem, ops);
  const i64 retained_input_bytes = [&]() {
    i64 bytes = 0;
    for (int tensor_id : external_inputs) {
      if (retained_inputs.count(tensor_id)) {
        bytes += problem.tensors[tensor_id].width * problem.tensors[tensor_id].height;
      }
    }
    return bytes;
  }();

  i64 compute_cost = 0;
  for (int op_index : ops) {
    compute_cost += problem.ops[op_index].base_cost;
  }

  const std::vector<i64> widths = SpatialCandidates(output.width, problem.native_width);
  const std::vector<i64> heights =
      SpatialCandidates(output.height, problem.native_height);

  for (i64 w : widths) {
    for (i64 h : heights) {
      const i64 tile_count_w = CeilDiv(output.width, w);
      const i64 tile_count_h = CeilDiv(output.height, h);

      double latency = 0.0;
      bool feasible = true;
      for (int tile_r = 0; tile_r < tile_count_h && feasible; ++tile_r) {
        const i64 tile_h = std::min(h, output.height - tile_r * h);
        for (int tile_c = 0; tile_c < tile_count_w; ++tile_c) {
          const i64 tile_w = std::min(w, output.width - tile_c * w);
          i64 external_bytes = 0;
          for (int tensor_id : external_inputs) {
            if (!retained_inputs.count(tensor_id)) {
              external_bytes += tile_h * tile_w;
            }
          }
          const i64 output_bytes = tile_h * tile_w;
          const i64 reserved_output_bytes =
              retain_final_output ? output.width * output.height : output_bytes;
          const i64 working_set =
              retained_input_bytes + external_bytes + reserved_output_bytes;
          if (working_set > problem.fast_memory_capacity) {
            feasible = false;
            break;
          }
          latency += std::max(static_cast<double>(compute_cost),
                              BytesToTime(external_bytes +
                                              (retain_final_output ? 0 : output_bytes),
                                          problem.slow_memory_bandwidth));
        }
      }
      if (!feasible) {
        continue;
      }
      if (latency < best.latency) {
        best.granularity = Granularity{w, h, 1};
        best.traversal_order = std::nullopt;
        best.tensors_to_retain =
            retain_final_output ? std::vector<int>{*output_tensor_id}
                                : std::vector<int>{};
        best.latency = latency;
      }
    }
  }
  return best;
}

double EvaluateMatmulNoSplitWithTraversal(
    const Problem& problem, const Op& matmul_op,
    const std::vector<int>& extra_pointwise_inputs, double extra_compute_cost,
    const std::unordered_set<int>& retained_inputs, bool retain_final_output,
    i64 output_width, i64 output_height, i64 w, i64 h,
    const std::vector<std::pair<int, int>>& traversal) {
  const Tensor& lhs = problem.tensors[matmul_op.inputs[0]];
  const Tensor& rhs = problem.tensors[matmul_op.inputs[1]];
  (void)rhs;
  const i64 k_dim = lhs.width;
  const double compute_cost =
      static_cast<double>(matmul_op.base_cost) *
          static_cast<double>(k_dim) /
          static_cast<double>(problem.native_width) +
      extra_compute_cost;

  const i64 rows = CeilDiv(output_height, h);
  const i64 cols = CeilDiv(output_width, w);
  if (static_cast<i64>(traversal.size()) != rows * cols) {
    return kInf;
  }

  const bool lhs_retained = retained_inputs.count(matmul_op.inputs[0]) != 0;
  const bool rhs_retained = retained_inputs.count(matmul_op.inputs[1]) != 0;
  i64 retained_input_bytes = 0;
  if (lhs_retained) {
    retained_input_bytes += lhs.width * lhs.height;
  }
  if (rhs_retained) {
    retained_input_bytes += rhs.width * rhs.height;
  }
  for (int tensor_id : extra_pointwise_inputs) {
    if (retained_inputs.count(tensor_id)) {
      retained_input_bytes +=
          problem.tensors[tensor_id].width * problem.tensors[tensor_id].height;
    }
  }
  const i64 reserved_output_bytes =
      retain_final_output ? output_width * output_height : 0;

  int cached_row = -1;
  int cached_col = -1;
  double total_latency = 0.0;

  for (const auto& [row, col] : traversal) {
    const i64 tile_h = std::min(h, output_height - static_cast<i64>(row) * h);
    const i64 tile_w = std::min(w, output_width - static_cast<i64>(col) * w);
    const i64 row_bytes = tile_h * k_dim;
    const i64 col_bytes = k_dim * tile_w;
    const i64 output_bytes = tile_h * tile_w;
    i64 pointwise_bytes = 0;
    for (int tensor_id : extra_pointwise_inputs) {
      if (!retained_inputs.count(tensor_id)) {
        pointwise_bytes += output_bytes;
      }
    }
    const i64 working_set =
        retained_input_bytes + (lhs_retained ? 0 : row_bytes) +
        (rhs_retained ? 0 : col_bytes) + pointwise_bytes +
        (retain_final_output ? reserved_output_bytes : output_bytes);
    if (working_set > problem.fast_memory_capacity) {
      return kInf;
    }

    i64 load_bytes = pointwise_bytes;
    if (!lhs_retained && cached_row != row) {
      load_bytes += row_bytes;
    }
    if (!rhs_retained && cached_col != col) {
      load_bytes += col_bytes;
    }
    total_latency +=
        std::max(compute_cost,
                 BytesToTime(load_bytes + (retain_final_output ? 0 : output_bytes),
                             problem.slow_memory_bandwidth));
    cached_row = row;
    cached_col = col;
  }
  return total_latency;
}

Plan EvaluateMatmulWithOptionalPointwiseTail(const Problem& problem,
                                             const std::vector<int>& ops,
                                             bool allow_tail,
                                             const std::unordered_set<int>& retained_inputs,
                                             bool retain_final_output) {
  Plan best;
  best.ops = ops;

  const Op& root = problem.ops[ops.front()];
  if (!IsMatMul(root) || root.inputs.size() != 2) {
    return best;
  }
  const std::optional<int> output_tensor_id = SingleOutputTensor(problem, ops);
  if (!output_tensor_id.has_value()) {
    return best;
  }
  const Tensor& output = problem.tensors[*output_tensor_id];
  const Tensor& lhs = problem.tensors[root.inputs[0]];
  const i64 k_dim = lhs.width;

  i64 pointwise_compute = 0;
  std::vector<int> external_inputs = ExternalInputs(problem, ops);
  if (!allow_tail) {
    external_inputs = ExternalInputs(problem, {ops.front()});
  }
  std::vector<int> extra_pointwise_inputs;
  for (int tensor_id : external_inputs) {
    if (tensor_id != root.inputs[0] && tensor_id != root.inputs[1]) {
      extra_pointwise_inputs.push_back(tensor_id);
    }
  }
  if (allow_tail) {
    for (std::size_t i = 1; i < ops.size(); ++i) {
      pointwise_compute += problem.ops[ops[i]].base_cost;
    }
  }

  const std::vector<i64> widths = SpatialCandidates(output.width, problem.native_width);
  const std::vector<i64> heights =
      SpatialCandidates(output.height, problem.native_height);

  for (i64 w : widths) {
    for (i64 h : heights) {
      if (allow_tail) {
        const auto raster = RasterOrder(CeilDiv(output.height, h), CeilDiv(output.width, w));
        const auto snake = SnakeOrder(CeilDiv(output.height, h), CeilDiv(output.width, w));
        const double raster_latency = EvaluateMatmulNoSplitWithTraversal(
            problem, root, extra_pointwise_inputs,
            static_cast<double>(pointwise_compute), retained_inputs,
            retain_final_output, output.width, output.height, w, h, raster);
        const double snake_latency = EvaluateMatmulNoSplitWithTraversal(
            problem, root, extra_pointwise_inputs,
            static_cast<double>(pointwise_compute), retained_inputs,
            retain_final_output, output.width, output.height, w, h, snake);
        double latency = raster_latency;
        std::optional<std::vector<int>> traversal;
        if (snake_latency < latency) {
          latency = snake_latency;
          traversal = FlattenTraversalOrder(snake, CeilDiv(output.width, w));
        }
        if (!std::isfinite(latency)) {
          continue;
        }
        if (latency < best.latency) {
          best.granularity = Granularity{w, h, k_dim};
          best.traversal_order = traversal;
          best.tensors_to_retain =
              retain_final_output ? std::vector<int>{*output_tensor_id}
                                  : std::vector<int>{};
          best.latency = latency;
        }
        continue;
      }

      for (i64 k : KCandidates(k_dim, problem.native_width)) {
        const i64 rows = CeilDiv(output.height, h);
        const i64 cols = CeilDiv(output.width, w);

        if (k == k_dim) {
          const auto raster = RasterOrder(rows, cols);
          const auto snake = SnakeOrder(rows, cols);
          double raster_latency = EvaluateMatmulNoSplitWithTraversal(
              problem, root, {}, 0.0, retained_inputs, retain_final_output,
              output.width, output.height, w, h, raster);
          double snake_latency = EvaluateMatmulNoSplitWithTraversal(
              problem, root, {}, 0.0, retained_inputs, retain_final_output,
              output.width, output.height, w, h, snake);
          double latency = raster_latency;
          std::optional<std::vector<int>> traversal;
          if (snake_latency < latency) {
            latency = snake_latency;
            traversal = FlattenTraversalOrder(snake, cols);
          }
          if (!std::isfinite(latency)) {
            continue;
          }
          if (latency < best.latency) {
            best.granularity = Granularity{w, h, k};
            best.traversal_order = traversal;
            best.tensors_to_retain =
                retain_final_output ? std::vector<int>{*output_tensor_id}
                                    : std::vector<int>{};
            best.latency = latency;
          }
          continue;
        }

        const i64 red_steps = CeilDiv(k_dim, k);
        const bool lhs_retained = retained_inputs.count(root.inputs[0]) != 0;
        const bool rhs_retained = retained_inputs.count(root.inputs[1]) != 0;
        i64 retained_input_bytes = 0;
        if (lhs_retained) {
          retained_input_bytes += lhs.width * lhs.height;
        }
        if (rhs_retained) {
          retained_input_bytes +=
              problem.tensors[root.inputs[1]].width *
              problem.tensors[root.inputs[1]].height;
        }
        const i64 reserved_output_bytes =
            retain_final_output ? output.width * output.height : 0;
        double latency = 0.0;
        bool feasible = true;
        for (int row = 0; row < rows && feasible; ++row) {
          const i64 tile_h =
              std::min(h, output.height - static_cast<i64>(row) * h);
          for (int col = 0; col < cols; ++col) {
            const i64 tile_w =
                std::min(w, output.width - static_cast<i64>(col) * w);
            for (int step = 0; step < red_steps; ++step) {
              const i64 step_k =
                  std::min(k, k_dim - static_cast<i64>(step) * k);
              const i64 lhs_bytes = tile_h * step_k;
              const i64 rhs_bytes = step_k * tile_w;
              const i64 output_bytes = tile_h * tile_w;
              const i64 working_set =
                  retained_input_bytes + (lhs_retained ? 0 : lhs_bytes) +
                  (rhs_retained ? 0 : rhs_bytes) +
                  (retain_final_output ? reserved_output_bytes : output_bytes);
              if (working_set > problem.fast_memory_capacity) {
                feasible = false;
                break;
              }
              const i64 store_bytes =
                  (step + 1 == red_steps && !retain_final_output) ? output_bytes
                                                                  : 0;
              const double compute =
                  static_cast<double>(root.base_cost) *
                  static_cast<double>(step_k) /
                  static_cast<double>(problem.native_width);
              latency += std::max(
                  compute,
                  BytesToTime((lhs_retained ? 0 : lhs_bytes) +
                                  (rhs_retained ? 0 : rhs_bytes) + store_bytes,
                              problem.slow_memory_bandwidth));
            }
          }
        }
        if (!feasible) {
          continue;
        }
        if (latency < best.latency) {
          best.granularity = Granularity{w, h, k};
          best.traversal_order = std::nullopt;
          best.tensors_to_retain =
              retain_final_output ? std::vector<int>{*output_tensor_id}
                                  : std::vector<int>{};
          best.latency = latency;
        }
      }
    }
  }
  return best;
}

Plan EvaluatePlanForOps(const Problem& problem, const std::vector<int>& ops,
                        const std::unordered_set<int>& retained_inputs,
                        bool retain_final_output) {
  if (ops.empty()) {
    return Plan();
  }
  if (IsPointwise(problem.ops[ops.front()])) {
    return EvaluatePointwiseChain(problem, ops, retained_inputs,
                                  retain_final_output);
  }

  const bool allow_tail = ops.size() > 1;
  Plan candidate = EvaluateMatmulWithOptionalPointwiseTail(
      problem, ops, allow_tail, retained_inputs, retain_final_output);
  if (!std::isfinite(candidate.latency) && allow_tail) {
    return EvaluateMatmulWithOptionalPointwiseTail(
        problem, {ops.front()}, false, retained_inputs, retain_final_output);
  }
  return candidate;
}

Plan BestPlanForChain(const Problem& problem, const std::vector<int>& chain,
                      const std::unordered_set<int>& retained_inputs) {
  Plan best;
  for (std::size_t length = 1; length <= chain.size(); ++length) {
    std::vector<int> prefix(chain.begin(), chain.begin() + static_cast<long>(length));
    Plan candidate =
        EvaluatePlanForOps(problem, prefix, retained_inputs, false);
    if (candidate.Score() < best.Score() ||
        (candidate.Score() == best.Score() &&
         candidate.ops.size() > best.ops.size())) {
      best = candidate;
    }
  }
  return best;
}

int ChooseNextRoot(const Problem& problem, const std::vector<bool>& scheduled,
                   const std::vector<int>& ready_ops,
                   const std::unordered_set<int>& retained_inputs) {
  int best_root = ready_ops.front();
  double best_score = kInf;
  std::size_t best_length = 0;

  for (int root : ready_ops) {
    const std::vector<int> chain = BuildMaxChain(problem, scheduled, root);
    const Plan plan = BestPlanForChain(problem, chain, retained_inputs);
    if (plan.ops.empty() || !std::isfinite(plan.latency)) {
      continue;
    }
    if (plan.Score() < best_score ||
        (plan.Score() == best_score && plan.ops.size() > best_length) ||
        (plan.Score() == best_score && plan.ops.size() == best_length &&
         root < best_root)) {
      best_score = plan.Score();
      best_root = root;
      best_length = plan.ops.size();
    }
  }
  return best_root;
}

std::vector<int> InitialReadyOps(const Problem& problem) {
  std::vector<int> ready;
  for (int op_index = 0; op_index < static_cast<int>(problem.ops.size()); ++op_index) {
    bool is_ready = true;
    for (int tensor_id : problem.ops[op_index].inputs) {
      if (problem.producer[tensor_id] != -1) {
        is_ready = false;
        break;
      }
    }
    if (is_ready) {
      ready.push_back(op_index);
    }
  }
  return ready;
}

std::vector<Plan> Solve(const Problem& problem) {
  const int n = static_cast<int>(problem.ops.size());
  std::vector<bool> scheduled(n, false);
  std::unordered_set<int> retained_inputs;
  std::vector<int> remaining_internal_inputs(n, 0);
  for (int op_index = 0; op_index < n; ++op_index) {
    int count = 0;
    for (int tensor_id : problem.ops[op_index].inputs) {
      if (problem.producer[tensor_id] != -1) {
        ++count;
      }
    }
    remaining_internal_inputs[op_index] = count;
  }

  std::vector<int> ready_ops = InitialReadyOps(problem);
  std::vector<Plan> schedule;
  schedule.reserve(problem.ops.size());

  int scheduled_count = 0;
  while (scheduled_count < n) {
    if (ready_ops.empty()) {
      throw std::runtime_error("No schedulable operations remain");
    }
    const int root =
        ChooseNextRoot(problem, scheduled, ready_ops, retained_inputs);
    const std::vector<int> chain = BuildMaxChain(problem, scheduled, root);
    Plan plan = BestPlanForChain(problem, chain, retained_inputs);
    if (plan.ops.empty() || !std::isfinite(plan.latency)) {
      throw std::runtime_error("Unable to find a feasible plan for ready op");
    }

    std::unordered_set<int> picked(plan.ops.begin(), plan.ops.end());
    std::vector<int> next_ready;
    next_ready.reserve(ready_ops.size() + plan.ops.size());
    for (int op_index : ready_ops) {
      if (!picked.count(op_index)) {
        next_ready.push_back(op_index);
      }
    }

    for (int op_index : plan.ops) {
      if (scheduled[op_index]) {
        throw std::runtime_error("Operation scheduled twice");
      }
      scheduled[op_index] = true;
      ++scheduled_count;
      for (int output_tensor : problem.ops[op_index].outputs) {
        for (int consumer : problem.consumers[output_tensor]) {
          if (scheduled[consumer] || picked.count(consumer)) {
            continue;
          }
          --remaining_internal_inputs[consumer];
          if (remaining_internal_inputs[consumer] == 0) {
            next_ready.push_back(consumer);
          }
        }
      }
    }

    const auto final_output = SingleOutputTensor(problem, plan.ops);
    Plan best_variant = EvaluatePlanForOps(problem, plan.ops, retained_inputs, false);
    std::unordered_set<int> next_retained_inputs;
    double best_objective = best_variant.latency;
    if (scheduled_count < n && !next_ready.empty()) {
      const int next_root =
          ChooseNextRoot(problem, scheduled, next_ready, next_retained_inputs);
      const std::vector<int> next_chain =
          BuildMaxChain(problem, scheduled, next_root);
      best_objective +=
          BestPlanForChain(problem, next_chain, next_retained_inputs).latency;
    }

    if (final_output.has_value()) {
      bool has_future_consumer = false;
      for (int consumer : problem.consumers[*final_output]) {
        if (!scheduled[consumer]) {
          has_future_consumer = true;
          break;
        }
      }
      if (has_future_consumer) {
        std::unordered_set<int> candidate_retained{*final_output};
        Plan retained_variant =
            EvaluatePlanForOps(problem, plan.ops, retained_inputs, true);
        if (std::isfinite(retained_variant.latency)) {
          double objective = retained_variant.latency;
          if (scheduled_count < n && !next_ready.empty()) {
            const int next_root = ChooseNextRoot(problem, scheduled, next_ready,
                                                 candidate_retained);
            const std::vector<int> next_chain =
                BuildMaxChain(problem, scheduled, next_root);
            const Plan next_plan =
                BestPlanForChain(problem, next_chain, candidate_retained);
            objective += next_plan.latency;
          }
          if (objective < best_objective) {
            best_objective = objective;
            best_variant = retained_variant;
            next_retained_inputs = std::move(candidate_retained);
          }
        }
      }
    }

    schedule.push_back(best_variant);
    retained_inputs = std::move(next_retained_inputs);

    std::sort(next_ready.begin(), next_ready.end());
    next_ready.erase(std::unique(next_ready.begin(), next_ready.end()),
                     next_ready.end());
    ready_ops = std::move(next_ready);
  }

  return schedule;
}

void ValidateSolution(const Problem& problem, const std::vector<Plan>& plans) {
  std::vector<bool> seen(problem.ops.size(), false);
  for (const Plan& plan : plans) {
    if (plan.ops.empty()) {
      throw std::runtime_error("Empty subgraph in solution");
    }
    if (!std::isfinite(plan.latency) || plan.latency <= 0.0) {
      throw std::runtime_error("Invalid subgraph latency");
    }
    for (int op_index : plan.ops) {
      if (op_index < 0 || op_index >= static_cast<int>(problem.ops.size())) {
        throw std::runtime_error("Operation index out of range");
      }
      seen[op_index] = true;
    }
  }
  for (bool present : seen) {
    if (!present) {
      throw std::runtime_error("Solution did not cover every operation");
    }
  }
}

void WriteSolution(const std::string& path, const std::vector<Plan>& plans) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Unable to open output file: " + path);
  }

  out << "{\n";

  out << "  \"subgraphs\": [\n";
  for (std::size_t i = 0; i < plans.size(); ++i) {
    out << "    [";
    for (std::size_t j = 0; j < plans[i].ops.size(); ++j) {
      if (j != 0) {
        out << ", ";
      }
      out << plans[i].ops[j];
    }
    out << "]";
    out << (i + 1 == plans.size() ? "\n" : ",\n");
  }
  out << "  ],\n";

  out << "  \"granularities\": [\n";
  for (std::size_t i = 0; i < plans.size(); ++i) {
    const Granularity& g = plans[i].granularity;
    out << "    [" << g.width << ", " << g.height << ", " << g.depth << "]";
    out << (i + 1 == plans.size() ? "\n" : ",\n");
  }
  out << "  ],\n";

  out << "  \"tensors_to_retain\": [\n";
  for (std::size_t i = 0; i < plans.size(); ++i) {
    out << "    [";
    for (std::size_t j = 0; j < plans[i].tensors_to_retain.size(); ++j) {
      if (j != 0) {
        out << ", ";
      }
      out << plans[i].tensors_to_retain[j];
    }
    out << "]";
    out << (i + 1 == plans.size() ? "\n" : ",\n");
  }
  out << "  ],\n";

  out << "  \"traversal_orders\": [\n";
  for (std::size_t i = 0; i < plans.size(); ++i) {
    if (!plans[i].traversal_order.has_value()) {
      out << "    null";
    } else {
      out << "    [";
      const auto& order = *plans[i].traversal_order;
      for (std::size_t j = 0; j < order.size(); ++j) {
        if (j != 0) {
          out << ", ";
        }
        out << order[j];
      }
      out << "]";
    }
    out << (i + 1 == plans.size() ? "\n" : ",\n");
  }
  out << "  ],\n";

  out << "  \"subgraph_latencies\": [\n";
  out << std::fixed << std::setprecision(6);
  for (std::size_t i = 0; i < plans.size(); ++i) {
    out << "    " << plans[i].latency;
    out << (i + 1 == plans.size() ? "\n" : ",\n");
  }
  out << "  ]\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "Usage: " << argv[0] << " <input.json> <output.json>\n";
      return 1;
    }

    const Problem problem = ReadProblem(argv[1]);
    const std::vector<Plan> plans = Solve(problem);
    ValidateSolution(problem, plans);
    WriteSolution(argv[2], plans);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "mlsys error: " << ex.what() << '\n';
    return 1;
  }
}
