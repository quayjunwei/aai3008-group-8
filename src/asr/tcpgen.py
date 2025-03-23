# tcpgen.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperTokenizer
from typing import List

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
        super(TCPGen, self).__init__()
        self.trie = Trie()
        self.tokenizer = tokenizer
        self.biasing_subword_to_idx = {}
        self._load_biasing_lists(biasing_files)
        self.decoder_state_dim = decoder_state_dim
        self.embedding_dim = embedding_dim

        num_subwords = len(self.biasing_subword_to_idx)
        self.bias_embeddings = nn.Embedding(num_subwords, self.embedding_dim)

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
                            for subword_id in subword_ids:
                                # if it’s not already in the dictionary, add it
                                if subword_id not in self.biasing_subword_to_idx:
                                    # assign it the next index in the embedding table
                                    self.biasing_subword_to_idx[subword_id] = len(self.biasing_subword_to_idx)

            except Exception as e:
                print(f"Error reading file {file}: {e}")

    # def get_valid_tokens(self, prefix_ids: List[int]) -> List[int]:
    #     return self.trie.starts_with(prefix_ids)
    
    def _get_bias_embedding(self, token_id: int) -> torch.Tensor:
        if token_id in self.biasing_subword_to_idx:
            idx = self.biasing_subword_to_idx[token_id]
            return self.bias_embeddings(torch.tensor(idx, dtype=torch.long))
        else:
            return torch.zeros(self.embedding_dim, device=self.bias_embeddings.weight.device)

    def forward(self, original_logits, prefix, decoder_state, threshold=0.5) -> torch.Tensor:
        batch_size, vocab_size = original_logits.size()

        # For each item in batch, get valid next subword IDs.
        valid_tokens_batch = []
        for b in range(batch_size):
            prefix_ids = prefix[b].tolist()
            valid_ids = self.trie.starts_with(prefix_ids)
            valid_tokens_batch.append(valid_ids)

        original_probs = F.softmax(original_logits, dim=-1)  # [B, V]
        pointer_dists = []

        for b in range(batch_size):
            valid_ids = valid_tokens_batch[b]

            # If no valid tokens, pointer distribution = original distribution
            if not valid_ids:
                pointer_dists.append(original_probs[b])
                continue

            # For each valid token ID, get an embedding & pointer score
            dec_state_expanded = decoder_state[b].unsqueeze(0).expand(len(valid_ids), -1)

            pointer_scores = []
            for token_id in valid_ids:
                token_emb = self._get_bias_embedding(token_id)
                concat_vec = torch.cat([dec_state_expanded[0], token_emb], dim=-1)
                score = self.pointer_net(concat_vec)
                pointer_scores.append(score)

            # Convert pointer scores to a distribution over just valid_ids
            pointer_scores_tensor = torch.stack(pointer_scores, dim=0).squeeze(-1)
            pointer_probs = F.softmax(pointer_scores_tensor, dim=0)

            # Scatter those pointer_probs back into a [V]-sized vector
            pointer_dist = torch.zeros_like(original_probs[b])
            for i, token_id in enumerate(valid_ids):
                pointer_dist[token_id] = pointer_probs[i]

            pointer_dists.append(pointer_dist)

        # pointer_dists: list of length B, each a [V] vector
        pointer_dists = torch.stack(pointer_dists, dim=0)

        # Compute generation probability p_gen from decoder_state
        # p_gen_net outputs shape [B, 1] in [0,1]
        p_gen = self.p_gen_net(decoder_state)
        p_gen = p_gen.expand(-1, vocab_size)

        # Combine pointer distribution & original distribution
        biased_probs = p_gen * pointer_dists + (1.0 - p_gen) * original_probs

        # If sum of pointer_dist < threshold, revert to original_probs
        pointer_mass = pointer_dists.sum(dim=-1, keepdim=True)
        # shape for broadcast: [B,V]
        final_probs = torch.where(
            pointer_mass < threshold,
            original_probs,
            biased_probs
        )

        # Convert back to logits for subsequent cross-entropy
        final_logits = torch.log(final_probs + 1e-8)
        return final_logits
