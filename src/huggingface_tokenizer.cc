
/*!
 *  Copyright (c) 2023 by Contributors
 * \file huggingface_tokenizer.cc
 * \brief Huggingface tokenizer
 */
#include <huggingface_tokenizer.h>
#include <tokenizers_c.h>

#include <cassert>
#include <cstdint>

namespace tokenizers {
/*!
 * \brief A simple c++ header of tokenizer via C API.
 */

HFTokenizer::HFTokenizer(TokenizerHandle handle) : handle_(handle) {
#ifdef COMPILE_WASM_RUNTIME
  setenv("TOKENIZERS_PARALLELISM", "false", true);
#endif
}

HFTokenizer::~HFTokenizer() {
  if (handle_ != nullptr) {
    tokenizers_free(handle_);
  }
}

// use i32 to be consistent with sentencepiece
HFEncoding HFTokenizer::Encode(const std::string& text, bool add_special_tokens) {
  TokenizerEncodeResult result;
  tokenizers_encode(handle_, text.data(), text.length(), static_cast<int>(add_special_tokens),
                    &result);
  std::vector<int32_t> ids(result.token_ids, result.token_ids + result.len);
  std::vector<int32_t> attention_mask(result.attention_mask, result.attention_mask + result.len);
  tokenizers_free_encode_results(&result, 1);
  auto ret = HFEncoding{
      ids,
      attention_mask,
  };
  return ret;
}

// use i32 to be consistent with sentencepiece
HFEncoding HFTokenizer::Encode(const std::string& text) { return Encode(text, false); }

std::vector<HFEncoding> HFTokenizer::EncodeBatch(const std::vector<std::string>& texts,
                                                 bool add_special_tokens) {
  std::vector<const char*> texts_raw;
  std::vector<size_t> seq_lens;
  size_t num_seqs = texts.size();
  texts_raw.reserve(num_seqs);
  seq_lens.reserve(num_seqs);
  for (const auto& text : texts) {
    texts_raw.push_back(text.data());
    seq_lens.push_back(text.length());
  }
  std::vector<TokenizerEncodeResult> results(num_seqs);
  tokenizers_encode_batch(handle_, texts_raw.data(), seq_lens.data(), texts.size(),
                          static_cast<int>(add_special_tokens), results.data());
  std::vector<HFEncoding> ret;
  ret.reserve(texts.size());
  for (size_t i = 0; i < texts.size(); ++i) {
    std::vector<int32_t> ids(results[i].token_ids, results[i].token_ids + results[i].len);
    std::vector<int32_t> attention_mask(results[i].attention_mask,
                                        results[i].attention_mask + results[i].len);
    auto cur = HFEncoding{
        ids,
        attention_mask,
    };
    ret.push_back(cur);
  }
  tokenizers_free_encode_results(results.data(), texts.size());
  return ret;
}

std::vector<HFEncoding> HFTokenizer::EncodeBatch(const std::vector<std::string>& texts) {
  return EncodeBatch(texts, false);
}

// use i32 to be consistent with sentencepiece
std::string HFTokenizer::Decode(const std::vector<int32_t>& ids, bool skip_special_tokens) {
  tokenizers_decode(handle_, reinterpret_cast<const uint32_t*>(ids.data()), ids.size(),
                    static_cast<int>(skip_special_tokens));
  const char* data;
  size_t len;
  tokenizers_get_decode_str(handle_, &data, &len);
  return std::string(data, len);
}

std::string HFTokenizer::Decode(const std::vector<int32_t>& ids) { return Decode(ids, false); }

size_t HFTokenizer::GetVocabSize() {
  size_t size;
  tokenizers_get_vocab_size(handle_, &size);
  assert(size > 0);
  return size;
}

std::string HFTokenizer::IdToToken(int32_t id) {
  const char* data;
  size_t len;
  tokenizers_id_to_token(handle_, static_cast<uint32_t>(id), &data, &len);
  return std::string(data, len);
}

int32_t HFTokenizer::TokenToId(const std::string& token) {
  int32_t id;
  tokenizers_token_to_id(handle_, token.data(), token.length(), &id);
  return id;
}

std::unique_ptr<HFTokenizer> HFTokenizer::FromBlobJSON(const std::string& json) {
  return std::make_unique<HFTokenizer>(tokenizers_new_from_str(json.data(), json.length()));
}

}  // namespace tokenizers
