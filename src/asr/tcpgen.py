# tcpgen.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperTokenizer

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def starts_with(self, prefix: str) -> list:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        results = []
        self._dfs(node, prefix, results)
        return results

    def _dfs(self, node: TrieNode, prefix: str, results: list):
        if node.is_end:
            results.append(prefix)
        for char, child in node.children.items():
            self._dfs(child, prefix + char, results)

class TCPGen(nn.Module):
    def __init__(self, biasing_files: list, tokenizer=WhisperTokenizer, decoder_state_dim=256, embedding_dim=64):
        """
        Initializes the TCPGen module.
        
        Parameters:
          biasing_files: List of file paths containing biasing vocabulary.
          decoder_state_dim: Dimension of the decoder state vector.
          embedding_dim: Dimension for learned token embeddings.
        """
        super(TCPGen, self).__init__()
        self.trie = Trie()
        self.tokenizer = tokenizer
        self.biasing_words = set()
        self._load_biasing_lists(biasing_files)
        self.decoder_state_dim = decoder_state_dim
        self.embedding_dim = embedding_dim

        # Build a mapping from biasing words to indices (sorted for consistency)
        self.biasing_word_to_idx = {word: i for i, word in enumerate(sorted(self.biasing_words))}

        # Create an embedding table for biasing words
        self.bias_embeddings = nn.Embedding(len(self.biasing_words), embedding_dim)

        # Pointer network: takes concatenated [decoder_state, token_embedding] and outputs a score.
        self.pointer_net = nn.Sequential(
            nn.Linear(decoder_state_dim + embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Generation probability network: outputs p_gen in [0,1] given the decoder state.
        self.p_gen_net = nn.Sequential(
            nn.Linear(decoder_state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def _load_biasing_lists(self, biasing_files: list):
        for file in biasing_files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    for line in f:
                        word = line.strip()
                        if word:
                            subword_ids = self.tokenizer(word).input_ids
                            self.trie.insert(subword_ids)
                            self.biasing_words.add(word)
            except Exception as e:
                print(f"Error reading file {file}: {e}")

    def get_valid_tokens(self, prefix: str) -> list:
        return self.trie.starts_with(prefix)
    
    def get_token_embedding(self, token: str):
        """
        Retrieves the learned embedding for a given token.
        If the token is not in the biasing list, returns a zero vector.
        """
        if token in self.biasing_word_to_idx:
            idx = self.biasing_word_to_idx[token]
            return self.bias_embeddings(torch.tensor(idx, dtype=torch.long))
        else:
            return torch.zeros(self.embedding_dim)

    def forward(self, original_logits, prefix, decoder_state, threshold=0.5) -> dict:
        """
        Applies the tree-constrained pointer generator to bias the original logits.
        
        Parameters:
          original_logits: Logits of last whisper_model forward pass. (whisper_outputs.logits[:, -1, :])
          prefix: The current token prefix. (decoder_input_ids)
          decoder_state (torch.Tensor): The decoder state vector (whisper_outputs.last_hidden_state).
          threshold: Confidence threshold; if the bias distribution is too low, the original logits are returned.
          
        Returns:
          final_logits: New logits after applying TCPGen biasing.
        """
        valid_tokens = list(self.get_valid_tokens(prefix))
        if not valid_tokens:
            return original_logits
        
        # Obtain embeddings for valid tokens.
        valid_token_embeddings = []
        for token in valid_tokens:
            emb = self.get_token_embedding(token)
            valid_token_embeddings.append(emb)
        valid_token_embeddings = torch.stack(valid_token_embeddings)  # (num_valid, embedding_dim)
        
        # Expand decoder_state to match the number of valid tokens.
        decoder_state_expanded = decoder_state.unsqueeze(0).expand(valid_token_embeddings.size(0), -1)  # (num_valid, decoder_state_dim)
        
        # Concatenate and compute pointer scores.
        concat_features = torch.cat([decoder_state_expanded, valid_token_embeddings], dim=1)  # (num_valid, decoder_state_dim+embedding_dim)
        pointer_logits = self.pointer_net(concat_features).squeeze(-1)  # (num_valid,)
        pointer_probs = torch.softmax(pointer_logits, dim=0)  # distribution over valid tokens
        
        # Build a bias logits distribution over the full vocabulary.
        tokens = list(original_logits.keys())
        bias_logits = {}
        for token in tokens:
            if token in valid_tokens:
                idx = valid_tokens.index(token)
                # Incorporate pointer probability by adding the log probability as a bias.
                bias_logits[token] = original_logits[token] + torch.log(pointer_probs[idx] + 1e-10).item()
            else:
                bias_logits[token] = float('-inf')
        
        # Compute generation probability from the decoder state.
        p_gen = self.p_gen_net(decoder_state).item()
        
        # Convert logits to probability distributions.
        tokens_list = tokens
        orig_tensor = torch.tensor([original_logits[t] for t in tokens_list])
        bias_tensor = torch.tensor([bias_logits[t] for t in tokens_list])
        orig_probs = torch.softmax(orig_tensor, dim=0)
        bias_probs = torch.softmax(bias_tensor, dim=0)
        
        if torch.max(bias_probs).item() < threshold:
            return original_logits
        
        # Interpolate between the bias and original distributions.
        combined_probs = p_gen * bias_probs + (1 - p_gen) * orig_probs
        final_logits_tensor = torch.log(combined_probs + 1e-10)
        final_logits = {token: final_logits_tensor[i].item() for i, token in enumerate(tokens_list)}
        return final_logits
