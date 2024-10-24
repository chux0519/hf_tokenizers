#include <tokenizers_c.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tokenizers {

struct HFEncoding {
  // use i32 to be consistent with sentencepiece
  std::vector<int32_t> ids;
  std::vector<int32_t> attention_mask;
};

struct AddedToken {
  std::string content;
  int32_t id;
};

/*!
 * \brief A simple c++ header of tokenizer via C API.
 */
class HFTokenizer {
 public:
  explicit HFTokenizer(TokenizerHandle handle);

  HFTokenizer(const HFTokenizer&) = delete;
  HFTokenizer(HFTokenizer&& other) { std::swap(other.handle_, handle_); }

  ~HFTokenizer();

  HFEncoding Encode(const std::string& text, bool add_special_tokens);

  HFEncoding Encode(const std::string& text);

  std::vector<HFEncoding> EncodeBatch(const std::vector<std::string>& texts,
                                      bool add_special_tokens);

  std::vector<HFEncoding> EncodeBatch(const std::vector<std::string>& texts);

  std::string Decode(const std::vector<int32_t>& ids, bool skip_special_tokens);

  std::string Decode(const std::vector<int32_t>& ids);

  size_t GetVocabSize();

  std::vector<AddedToken> GetAddedTokens();

  std::string IdToToken(int32_t id);

  int32_t TokenToId(const std::string& token);

  static std::unique_ptr<HFTokenizer> FromBlobJSON(const std::string& json_blob);

 private:
  TokenizerHandle handle_{nullptr};
  std::vector<AddedToken> added_tokens;
  void InitAddedTokens();
};
}  // namespace tokenizers
