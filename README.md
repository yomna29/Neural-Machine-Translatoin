
# Neural Machine Translation with a Transformer Model
This repository contains the solution for an assignment focused on implementing and training a sequence-to-sequence (seq2seq) model based on the Transformer architecture for Neural Machine Translation (NMT). The model is trained on a parallel corpus of French (source) and English (target) sentences.

## Setup and InstallationThe notebook uses PyTorch and Hugging Face libraries for data handling and tokenization.

To set up the environment, you need to install the following packages:

```bash
# PyTorch is assumed to be installed, but can be uncommented if needed
# !pip install torch==1.8.0

# Install Huggingface transformers and datasets
!pip install transformers==4.27.0
!pip install datasets==2.10.0
!pip install -U datasets  # Upgrade datasets for full functionality

```

##💾 DataThe NMT data consists of parallel sentence pairs and is loaded using the Hugging Face `datasets` library from local disk resources.

| Split | Features | Number of Rows |
| --- | --- | --- |
| `train` | `['text_en', 'text_fr']` | 8701 |
| `validation` | `['text_en', 'text_fr']` | 485 |
| `test` | `['text_en', 'text_fr']` | 486 |

Pre-trained Byte-Pair Encoding (BPE) tokenizers are provided for both French (source) and English (target) languages to handle subword units.

## Model ArchitectureThe model is an Encoder-Decoder Transformer built from scratch (without using PyTorch's high-level `nn.Transformer` modules).

###Core Components Implemented* **`MultiHeadAttention`**: A flexible module supporting:
* Full self-attention (in the encoder).
* Causal masked self-attention (in the decoder).
* Cross-attention (in the decoder, attending to encoder outputs).


* **`TransformerEmbeddings`**: Combines token embeddings with learned positional embeddings. It also includes the `compute_logits` function for projecting decoder output back to the vocabulary space.
* **`TransformerBlock`**: A single encoder or decoder layer incorporating attention, residual connections, layer normalization, and a feedforward network.
* **`EncoderDecoderModel`**: The main seq2seq model composed of:
* Source and target embedding layers.
* An `nn.ModuleList` of `num_encoder_layers` for the encoder.
* An `nn.ModuleList` of `num_decoder_layers` for the decoder.



###Training ConfigurationThe model was trained for **15 epochs** using the following hyperparameters:

* **Model Size (Hidden Size)**: 32
* **Intermediate Size (Feedforward)**: 128
* **Number of Attention Heads**: 4
* **Number of Encoder Layers**: 3
* **Number of Decoder Layers**: 3
* **Max Sequence Length**: 32
* **Dropout Probability**: 0.1
* **Optimizer**: Adam with a learning rate of 10^{-3}.
* **Total Trainable Parameters**: 401,536

##📈 Results and Evaluation###Training MetricsThe model was trained for 15 epochs, and the **best model checkpoint** was saved based on the lowest perplexity on the validation set.

* **Final Validation Perplexity**: **4.9989** (after step 4000)

###Test Set EvaluationThe model was evaluated on the test set using **beam search** decoding with a `beam_width` of 5.

* **Corpus BLEU Score**: **54.78** (or `0.547786...`)

###Sample TranslationsThe translation quality is generally good, especially for simple sentences.

* **Correct Translations**: Samples 11, 16, 17 (`you re the teacher .`, `i m leaving today .`, `i m no saint .`)
* **Minor Errors/Different Phrasing**: Samples 10, 13, 14, 15, 18.
* **Incorrect/Ungrammatical**: Sample 12 (`elle sourit avec bonheur .` -> `she s getting good him .`)

A common mistake observed is the incorrect translation of words immediately following common phrases like "she," "you're," "I'm not," which suggests difficulties with precise lexical choice or complex idiomatic expressions, especially given the small model size and limited vocabulary.

##📝 Answers to Assignment Questions###(b) Model Details* **(i) Vocabulary Size**: The vocabulary size used for both the source (French) and target (English) languages is **3200**.
* **(ii) Average Batch Statistics (Training)**:
* Average source tokens per sequence: **13.4**
* Average target tokens per sequence: **11.7**
* Proportion of padding in source: **32.9%**
* Proportion of padding in target: **29.2%**


* **(iii) Purpose of `model.pt**`: The model parameters are saved in `model.pt` during training whenever the model achieves a **new best perplexity score on the validation set**. The purpose is to:
1. **Retrieve the Best Performing Model**: Ensure that the final model used for testing and deployment is the one that performed optimally on unseen data (validation set), preventing the use of an overfitted model.
2. **Flexibility and Robustness**: Allow training to be stopped and resumed, or to recover the best state if training is interrupted.



###(d) Beam Search Efficiency Issue and Fix**The Issue:**

The provided `beam_search` implementation is inefficient because it **recomputes the decoder's entire sequence output for every single beam candidate at every single decoding step**.

When calculating the log probabilities for the next token, the code calls `model.forward_decoder(generation, ...)` where `generation` is the *full* partial sequence generated so far (including the previous tokens whose logits were already computed). For a sequence of length L, this results in L redundant computations inside the decoder.

**The Proposed Fix (Efficient Decoding):**

To fix this, the decoder must be adapted to calculate the output **only for the last, newly generated token** at each time step. This requires the decoder to maintain and reuse its internal state (key and value caches) from previous steps.

1. **Modify Encoder Output**: The `model.forward_encoder` is already efficient as it runs only once. The encoder output (`encoder_outputs`) remains constant and is reused.
2. **Modify Decoder for Step-by-Step Generation with Caching**:
* The `EncoderDecoderModel` and `TransformerBlock` should be updated to optionally return and accept a **key/value cache** (`past_key_values`) for the self-attention and cross-attention modules.
* When the cache is provided, the attention mechanisms should:
* Prepend the cached keys/values to the newly computed keys/values.
* Run the forward pass only on the last (newest) input token to the decoder.


* The `model.forward_decoder` signature would change:
```python
def forward_decoder(self,
                    input_ids: torch.LongTensor, # Now often just the last token ID
                    encoder_outputs: torch.FloatTensor,
                    encoder_padding_mask: torch.BoolTensor,
                    past_key_values: Optional[List[Tuple]] = None
                    ) -> Tuple[torch.FloatTensor, List[Tuple]]:
# Returns (next_token_logits, new_past_key_values)

```




3. **Update Beam Search Loop**:
* In the `beam_search` function, the loop would then pass only the last token and the accumulated `past_key_values` from the highest-scoring beams.
* The expensive call `decoder_output = model.forward_decoder(generation, ...)` would be replaced by a call that processes a single token and reuses the cache.



By implementing this caching mechanism, the decoder avoids re-processing the entire previously generated sequence, making the beam search significantly more efficient for autoregressive generation.

