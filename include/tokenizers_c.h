/*!
 *  Copyright (c) 2023 by Contributors
 * \file tokenizers_c.h
 * \brief C binding to tokenizers rust library
 */
#ifndef TOKENIZERS_C_H_
#define TOKENIZERS_C_H_

// The C API
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef void* TokenizerHandle;

typedef struct {
  int* token_ids;
  int* attention_mask;
  size_t len;
} TokenizerEncodeResult;

TokenizerHandle tokenizers_new_from_str(const char* json, size_t len);

int tokenizers_iterate_added_vocab(TokenizerHandle handle,
                                   void (*callback)(const char*, uint32_t, void* user_data),
                                   void* user_data);

void tokenizers_encode(TokenizerHandle handle, const char* data, size_t len, int add_special_token,
                       TokenizerEncodeResult* result);

void tokenizers_encode_batch(TokenizerHandle handle, const char** data, size_t* len,
                             size_t num_seqs, int add_special_token,
                             TokenizerEncodeResult* results);

void tokenizers_free_encode_results(TokenizerEncodeResult* results, size_t num_seqs);

void tokenizers_decode(TokenizerHandle handle, const uint32_t* data, size_t len,
                       int skip_special_token);

void tokenizers_get_decode_str(TokenizerHandle handle, const char** data, size_t* len);

void tokenizers_get_vocab_size(TokenizerHandle handle, size_t* size);

void tokenizers_id_to_token(TokenizerHandle handle, uint32_t id, const char** data, size_t* len);

// tokenizers_token_to_id stores -1 to *id if the token is not in the vocab
void tokenizers_token_to_id(TokenizerHandle handle, const char* token, size_t len, int32_t* id);

void tokenizers_free(TokenizerHandle handle);

#ifdef __cplusplus
}
#endif
#endif  // TOKENIZERS_C_H_
