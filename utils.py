import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LogitsProcessor,
    BartForConditionalGeneration,
)
import hashlib
import math

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
model.eval()

# Align pad_token with eos_token
tokenizer.pad_token = tokenizer.eos_token

print(f"Model vocab size: {model.config.vocab_size}")    # Expected: 50264 or similar
print(f"Tokenizer vocab size: {len(tokenizer)}")       # Should match model's vocab size
print(f"Pad token ID: {tokenizer.pad_token_id}")        # Should match eos_token_id
print(f"EOS token ID: {tokenizer.eos_token_id}")        # Typically 2 for BART

# Define helper functions
def generate_seed(context_ids, key=b'secret_key'):
    context_bytes = '_'.join(map(str, context_ids)).encode('utf-8')
    combined = key + context_bytes
    hash_digest = hashlib.sha256(combined).digest()
    seed = int.from_bytes(hash_digest[:4], 'big')
    return seed

def get_green_red_lists(seed, vocab_size, gamma=0.4):
    torch.manual_seed(seed)
    shuffled_indices = torch.randperm(vocab_size)
    split_idx = int(gamma * vocab_size)
    green_list = shuffled_indices[:split_idx]
    red_list = shuffled_indices[split_idx:]
    return green_list, red_list

# Define the custom LogitsProcessor
class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(self, gamma=0.4, delta=5.0, key=b'secret_key', n=10, verbose=False):
        super().__init__()
        self.gamma = gamma
        self.delta = delta
        self.key = key
        self.n = n
        self.verbose = verbose

    def generate_seed(self, context_ids):
        return generate_seed(context_ids, self.key)

    def get_green_red_lists(self, seed, vocab_size):
        return get_green_red_lists(seed, vocab_size, self.gamma)

    def __call__(self, input_ids, scores):
        # Get vocab_size from scores
        vocab_size = scores.size(-1)

        # Get the last `n` tokens for context
        context_ids = input_ids[0][-self.n:].tolist() if input_ids.size(1) >= self.n else input_ids[0].tolist()

        # Generate seed from context
        seed = self.generate_seed(context_ids)

        # Get green and red lists
        green_list, _ = self.get_green_red_lists(seed, vocab_size)

        # Exclude special tokens from the green list
        special_token_ids = set(tokenizer.all_special_ids)
        green_list = [token_id for token_id in green_list if token_id not in special_token_ids]

        # Apply watermark bias to green tokens
        scores[:, green_list] += self.delta

        if self.verbose:
            finite_scores = scores[torch.isfinite(scores)]
            print(f"Logits after adding delta: min={finite_scores.min().item()}, max={finite_scores.max().item()}, mean={finite_scores.mean().item()}")

        return scores

def summarize_text(article, max_length, min_length, gamma=0.4, delta=5.0, key=b'secret_key', n=10, verbose=False):
    # Tokenize the article (encoder input)
    inputs = tokenizer([article], max_length=1024, truncation=True, return_tensors='pt')

    # Initialize the logits processor
    logits_processor = WatermarkLogitsProcessor(
        gamma=gamma,
        delta=delta,
        key=key,
        n=n,
        verbose=verbose
    )

    # Generate summary using the custom logits processor
    summary_ids = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        min_length=min_length,
        logits_processor=[logits_processor],  # Pass to logits_processor
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,      # Use sampling to allow for randomness
        num_beams=1,         # Use greedy decoding or sampling
        temperature=0.7,     # Adjust as needed
        # early_stopping=True,
        # length_penalty=1.0,
    )

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print(f"Total tokens generated: {summary_ids.size(1)}")

    return summary, summary_ids

def verify_watermark(generated_ids, gamma=0.4, key=b'secret_key', n=10, z_threshold=4.0, verbose=False):
    green_token_count = 0
    total_tokens = 0

    vocab_size = model.config.vocab_size

    generated_ids = generated_ids[0]  # Since generated_ids is 2D, we take the first (and only) batch

    for i in range(len(generated_ids) - 1):
        # Get context
        if i < n:
            context_ids = generated_ids[:i + 1].tolist()
        else:
            context_ids = generated_ids[i - n + 1:i + 1].tolist()

        seed = generate_seed(context_ids, key=key)
        green_list, _ = get_green_red_lists(seed, vocab_size, gamma)

        next_token_id = generated_ids[i + 1].item()

        # Skip invalid token IDs
        if next_token_id >= vocab_size:
            continue

        # Skip special tokens
        if next_token_id in tokenizer.all_special_ids:
            continue

        if next_token_id in green_list:
            green_token_count += 1
            if verbose:
                print(f"Token ID {next_token_id} at position {i+1} is in green_list.")
        else:
            if verbose:
                print(f"Token ID {next_token_id} at position {i+1} is NOT in green_list.")

        total_tokens += 1

    green_ratio = green_token_count / total_tokens if total_tokens > 0 else 0.0
    expected_ratio = gamma

    std = math.sqrt(expected_ratio * (1 - expected_ratio) / total_tokens) if total_tokens > 0 else 0.0
    z_score = (green_ratio - expected_ratio) / std if std != 0 else 0.0

    is_watermarked = z_score >= z_threshold

    return green_ratio, z_score, is_watermarked

# Generate summary without watermarking
def summarize_text_no_watermark(article, max_length=300, min_length=100):
    inputs = tokenizer([article], max_length=1024, truncation=True, return_tensors='pt')
    summary_ids = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        min_length=min_length,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        num_beams=1,
        no_repeat_ngram_size=3,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(f"Total tokens generated (No Watermark): {summary_ids.size(1)}")
    return summary, summary_ids

# # Generate and verify without watermark
# summary_no_watermark, summary_ids_no_watermark = summarize_text_no_watermark(
#     ARTICLE,
#     max_length=50,
#     min_length=30
# )
# print("\nSummary without Watermark:")
# print(summary_no_watermark)
#
# # Verify watermark without applying any bias
# green_ratio_no_wm, z_score_no_wm, is_watermarked_no_wm = verify_watermark(
#     summary_ids_no_watermark,
#     gamma=0.4,    # Consistent gamma
#     key=b'secret_key',
#     n=10,
#     z_threshold=4.0  # Ensure threshold is set correctly
# )
# print(f"\nGreen Token Ratio (No Watermark): {green_ratio_no_wm}")
# print(f"Z-Score (No Watermark): {z_score_no_wm}")
# print(f"Watermark Detected (No Watermark): {is_watermarked_no_wm}")
