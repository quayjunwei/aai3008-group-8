from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.generation.logits_process import LogitsProcessor
import torch
from tcpgen import TCPGen

class TCPGenLogitsProcessor(LogitsProcessor):
    def __init__(self, tcpgen, tokenizer):
        self.tcpgen = tcpgen
        self.tokenizer = tokenizer
        # Build an id-to-token mapping from the tokenizer's vocabulary.
        vocab = tokenizer.get_vocab()
        self.id_to_token = {id: token for token, id in vocab.items()}

    def __call__(self, input_ids, scores):
        # input_ids: Tensor of shape (batch_size, sequence_length)
        # scores: Tensor of shape (batch_size, vocab_size)
        new_scores = scores.clone()
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            # Decode the current sequence for each sample
            decoded_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            tokens = decoded_text.strip().split()
            current_prefix = tokens[-1] if tokens else ""
            
            token_logits = {}
            for j, score in enumerate(scores[i]):
                token = self.id_to_token.get(j, "")
                token_logits[token] = score.item()
            
            modified_logits = self.tcpgen.bias_logits(token_logits, current_prefix)
            
            # Convert modified logits back to a tensor
            modified_scores = [modified_logits.get(self.id_to_token.get(j, ""), float('-inf')) for j in range(len(scores[i]))]
            new_scores[i] = torch.tensor(modified_scores, device=scores.device)
        return new_scores

class ASRModel:
    def __init__(self, model_name="openai/whisper-tiny", biasing_files=None):
        if biasing_files is None:
            biasing_files = []
        self.model_name = model_name
        self.model, self.feature_extractor, self.tokenizer = self._load_model(model_name)
        self.tcpgen = TCPGen(biasing_files, self.tokenizer)

    # def _load_model(self, model_name):
    #     # Determine device (CUDA if available)
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    #     # Load the Whisper model from Hugging Face Transformers
    #     model = WhisperForConditionalGeneration.from_pretrained(model_name)
    #     model = model.to(device)
    #     # model.eval()
        
    #     # Load the WhisperProcessor which includes both the feature extractor and tokenizer
    #     processor = WhisperProcessor.from_pretrained(model_name)
    #     feature_extractor = processor.feature_extractor
    #     tokenizer = processor.tokenizer
        
    #     return model, feature_extractor, tokenizer
    
    def forward(self, raw_audio, sample_rate=16000, decoder_input_ids=None):
        """
        Forward pass for audio data. This function should preprocess the audio
        and pass it through the model to get the logits.
        """
        input_features = self.feature_extractor(
            raw_audio, 
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features
        whisper_outputs = self.model(input_features=input_features, decoder_input_ids=decoder_input_ids, return_dict=True)
        final_decoder_hidden_state = whisper_outputs.last_hidden_state
        whisper_logits = whisper_outputs.logits[:, -1, :]
        tcpgen_logits = self.tcpgen(original_logits=whisper_logits, prefix=decoder_input_ids, decoder_state=final_decoder_hidden_state)

    def transcribe_with_biasing(self, audio_path: str) -> str:
        """
        Transcribe the audio file using the Whisper model with TCPGen biasing.
        The audio is preprocessed using the feature extractor. During generation,
        a custom logits processor applies the TCPGen bias based on the current prefix.
        Returns the final transcription as a string.
        """
        # Preprocess audio using the feature extractor.
        # The feature extractor expects the audio file path; alternatively, you can load the audio with torchaudio.
        self.model.eval()
        inputs = self.feature_extractor(audio_path, return_tensors="pt").input_features
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        
        # Create our custom logits processor with TCPGen and tokenizer.
        logits_processor = [TCPGenLogitsProcessor(self.tcpgen, self.tokenizer)]
        
        # Generate token IDs using the model while applying our biasing processor.
        generated_ids = self.model.generate(inputs, logits_processor=logits_processor)
        
        # Decode the generated token IDs to get the transcription.
        transcription = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription
