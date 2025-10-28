// A simple C wrapper of tokenzier library
use core::panic;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::str::FromStr;
use tokenizers::tokenizer::{Tokenizer, Result};
use tokenizers::Encoding;

type IterateAddedVocabCallback = extern "C" fn(*const c_char, u32, *mut c_void);

pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
    decode_str: String,
    id_to_token_result: String,
}

#[repr(C)]
pub struct TokenizerEncodeResult {
    token_ids: *mut u32,
    attention_mask: *mut u32,
    len: usize,
}


impl FromStr for TokenizerWrapper {
    type Err = Box<dyn std::error::Error + Send + Sync>;

    fn from_str(s: &str) -> Result<Self> {
        Ok(TokenizerWrapper {
            tokenizer: Tokenizer::from_str(s).unwrap(),
            decode_str: String::new(),
            id_to_token_result: String::new(),
        })
    }
}

impl TokenizerWrapper {
    pub fn iterate_added_vocab(&self, callback: IterateAddedVocabCallback, user_data: *mut c_void) {
        let special_vocab = self.tokenizer.get_added_vocabulary().get_vocab();
        for (key, value) in special_vocab {
            let key_cstring = CString::new(key.as_str()).unwrap();
            callback(key_cstring.as_ptr(), *value, user_data);
        }
    }

    pub fn encode(&mut self, text: &str, add_special_tokens: bool) -> Encoding {
        
        self.tokenizer.encode(text, add_special_tokens).unwrap()
    }

    pub fn encode_batch(&mut self, texts: Vec<&str>, add_special_tokens: bool) -> Vec<Encoding> {
        let results = self
            .tokenizer
            .encode_batch(texts, add_special_tokens)
            .unwrap()
            .into_iter()
            .collect::<Vec<Encoding>>();
        self.tokenizer.get_added_vocabulary().get_vocab();
        results
    }

    pub fn decode(&mut self, ids: &[u32], skip_special_tokens: bool) {
        self.decode_str = self.tokenizer.decode(ids, skip_special_tokens).unwrap();
    }
}

#[no_mangle]
extern "C" fn tokenizers_new_from_str(input_cstr: *const u8, len: usize) -> *mut TokenizerWrapper {
    unsafe {
        let json = std::str::from_utf8(std::slice::from_raw_parts(input_cstr, len)).unwrap();
        Box::into_raw(Box::new(TokenizerWrapper::from_str(json).expect("Failed to create tokenizer from string")))
    }
}

#[no_mangle]
extern "C" fn tokenizers_iterate_added_vocab(
    handle: *mut TokenizerWrapper,
    callback: IterateAddedVocabCallback,
    user_data: *mut c_void,
) {
    unsafe { (*handle).iterate_added_vocab(callback, user_data) }
}

#[no_mangle]
extern "C" fn tokenizers_encode(
    handle: *mut TokenizerWrapper,
    input_cstr: *const u8,
    len: usize,
    add_special_tokens: i32,
    out_result: *mut TokenizerEncodeResult,
) {
    unsafe {
        let input_data = std::str::from_utf8(std::slice::from_raw_parts(input_cstr, len)).unwrap();
        let encoded = (*handle).encode(input_data, add_special_tokens != 0);
        let ids = encoded.get_ids().to_vec();
        let attention_mask: Vec<u32> = encoded.get_attention_mask().to_vec();

        if ids.len() != attention_mask.len() {
            panic!("ids and attention_mask should be the same length")
        }
        let len = ids.len();

        *out_result = TokenizerEncodeResult {
            token_ids: Box::into_raw(ids.into_boxed_slice()) as *mut u32,
            attention_mask: Box::into_raw(attention_mask.into_boxed_slice()) as *mut u32,
            len,
        };
    }
}

#[no_mangle]
extern "C" fn tokenizers_encode_batch(
    handle: *mut TokenizerWrapper,
    input_cstr: *const *const u8,
    input_len: *const usize,
    num_seqs: usize,
    add_special_tokens: i32,
    out_result: *mut TokenizerEncodeResult,
) {
    unsafe {
        let input_data = (0..num_seqs)
            .map(|i| {
                std::str::from_utf8(std::slice::from_raw_parts(
                    *input_cstr.add(i),
                    *input_len.add(i),
                ))
                .unwrap()
            })
            .collect::<Vec<&str>>();
        let encoded_batch = (*handle).encode_batch(input_data, add_special_tokens != 0);
        for (i, encoded) in encoded_batch.into_iter().enumerate() {
            let ids = encoded.get_ids().to_vec();
            let attention_mask: Vec<u32> = encoded.get_attention_mask().to_vec();
            let len = ids.len();
            if ids.len() != attention_mask.len() {
                panic!("ids and attention_mask should be the same length")
            }

            let result = TokenizerEncodeResult {
                token_ids: Box::into_raw(ids.into_boxed_slice()) as *mut u32,
                attention_mask: Box::into_raw(attention_mask.into_boxed_slice()) as *mut u32,
                len,
            };
            *out_result.add(i) = result;
        }
    }
}

#[no_mangle]
extern "C" fn tokenizers_free_encode_results(results: *mut TokenizerEncodeResult, num_seqs: usize) {
    unsafe {
        let slice = std::slice::from_raw_parts_mut(results, num_seqs);
        for result in &mut *slice {
            drop(Box::from_raw(std::slice::from_raw_parts_mut(
                result.token_ids,
                result.len,
            )));
            drop(Box::from_raw(std::slice::from_raw_parts_mut(
                result.attention_mask,
                result.len,
            )));
        }
    }
}

#[no_mangle]
extern "C" fn tokenizers_decode(
    handle: *mut TokenizerWrapper,
    input_ids: *const u32,
    len: usize,
    skip_special_tokens: i32,
) {
    unsafe {
        let input_data = std::slice::from_raw_parts(input_ids, len);
        (*handle).decode(input_data, skip_special_tokens != 0);
    }
}

#[no_mangle]
extern "C" fn tokenizers_get_decode_str(
    handle: *mut TokenizerWrapper,
    out_cstr: *mut *mut u8,
    out_len: *mut usize,
) {
    unsafe {
        *out_cstr = (*handle).decode_str.as_mut_ptr();
        *out_len = (&(*handle).decode_str).len();
    }
}

#[no_mangle]
extern "C" fn tokenizers_free(wrapper: *mut TokenizerWrapper) {
    unsafe {
        drop(Box::from_raw(wrapper));
    }
}

#[no_mangle]
extern "C" fn tokenizers_get_vocab_size(handle: *mut TokenizerWrapper, size: *mut usize) {
    unsafe {
        *size = (*handle).tokenizer.get_vocab_size(true);
    }
}

#[no_mangle]
extern "C" fn tokenizers_id_to_token(
    handle: *mut TokenizerWrapper,
    id: u32,
    out_cstr: *mut *mut u8,
    out_len: *mut usize,
) {
    unsafe {
        let str = (*handle).tokenizer.id_to_token(id);
        (*handle).id_to_token_result = str.unwrap_or_default();

        *out_cstr = (*handle).id_to_token_result.as_mut_ptr();
        *out_len = (&(*handle).id_to_token_result).len();
    }
}

#[no_mangle]
extern "C" fn tokenizers_token_to_id(
    handle: *mut TokenizerWrapper,
    token: *const u8,
    len: usize,
    out_id: *mut i32,
) {
    unsafe {
        let token: &str = std::str::from_utf8(std::slice::from_raw_parts(token, len)).unwrap();
        let id = (*handle).tokenizer.token_to_id(token);
        *out_id = match id {
            Some(id) => id as i32,
            None => -1,
        };
    }
}