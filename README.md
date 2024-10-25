# hf_tokenizers

This project is a modified version of [mlc-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp). The key difference is that this repository has removed support for sentencepiece, as it is tailored specifically for my `embeddings.cpp` project.
JavaScript support has not been tested and may not function properly at this time.

This project provides a cross-platform C++ tokenizer binding library that can be universally deployed.
It wraps and binds the [HuggingFace tokenizers library](https://github.com/huggingface/tokenizers).


## Getting Started

The easiest way is to add this project as a submodule and then
include it via `add_sub_directory` in your CMake project.
You also need to turn on `c++17` support.

- First, you need to make sure you have rust installed.
- If you are cross-compiling make sure you install the necessary target in rust.
  For example, run `rustup target add aarch64-apple-ios` to install iOS target.
- You can then link the library

See [example](example) folder for an example CMake project.

### Example Code

```c++
// - dist/tokenizer.json
void HuggingFaceTokenizerExample() {
  // Read blob from file.
  auto blob = LoadBytesFromFile("dist/tokenizer.json");
  // Note: all the current factory APIs takes in-memory blob as input.
  // This gives some flexibility on how these blobs can be read.
  auto tok = Tokenizer::FromBlobJSON(blob);
  std::string prompt = "What is the capital of Canada?";
  // call Encode to turn prompt into token ids
  std::vector<int> ids = tok->Encode(prompt);
  // call Decode to turn ids into string
  std::string decoded_prompt = tok->Decode(ids);
}
```

### Extra Details

Currently, the project generates three static libraries
- `libtokenizers_c.a`: the c binding to tokenizers rust library
- `libsentencepice.a`: sentencepiece static library
- `libtokenizers_cpp.a`: the cpp binding implementation

If you are using an IDE, you can likely first use cmake to generate
these libraries and add them to your development environment.
If you are using cmake, `target_link_libraries(yourlib tokenizers_cpp)`
will automatically links in the other two libraries.
You can also checkout [MLC LLM](https://github.com/mlc-ai/mlc-llm)
for as an example of complete LLM chat application integrations.

## Acknowledgements

This project is only possible thanks to the shoulders open-source ecosystems that we stand on.