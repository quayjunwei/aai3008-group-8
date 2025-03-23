from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.generation.logits_process import LogitsProcessor
import torch
from asr.tcpgen import TCPGen

class TCPGenLogitsProcessor(LogitsProcessor): # currently not using p_gen_net
    def __init__(self, tcpgen):
        """
        :param tcpgen: an instance of your TCPGen class,
                       which has a 'trie' attribute for checking valid tokens.
        """
        super().__init__()
        self.tcpgen = tcpgen

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        input_ids: [batch_size, current_sequence_length]
        scores:    [batch_size, vocab_size]
        Return updated scores where invalid tokens are set to -inf.
        """
        new_scores = scores.clone()
        batch_size, vocab_size = scores.shape

        for i in range(batch_size):
            prefix_ids = input_ids[i].tolist()  # numeric subword IDs
            valid_ids = self.tcpgen.trie.starts_with(prefix_ids)

            # Exclude all tokens not in valid_ids
            for token_id in range(vocab_size):
                if token_id not in valid_ids:
                    new_scores[i, token_id] = float('-inf')

        return new_scores

class ASRModel:
    def __init__(self, model_name="openai/whisper-tiny", biasing_files=None):
        if biasing_files is None:
            biasing_files = []
        self.model_name = model_name
        self.model, self.feature_extractor, self.tokenizer = self._load_model(model_name)
        self.tcpgen = TCPGen(biasing_files, self.tokenizer)

    def _load_model(self, model_name):
        # Determine device (CUDA if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the Whisper model from Hugging Face Transformers
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model = model.to(device)
        # model.eval()
        
        # Load the WhisperProcessor which includes both the feature extractor and tokenizer
        processor = WhisperProcessor.from_pretrained(model_name)
        feature_extractor = processor.feature_extractor
        tokenizer = processor.tokenizer
        
        return model, feature_extractor, tokenizer
    
    def forward(self, raw_audio, sample_rate=16000, decoder_input_ids=None):
        """
        Forward pass for audio data. This function should preprocess the audio
        and pass it through the model to get the logits.
        """
        input = self.feature_extractor(
            raw_audio, 
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features
        # input = input.to(next(self.model.parameters()).device)

        whisper_outputs = self.model(input_features=input, decoder_input_ids=decoder_input_ids, return_dict=True)
        final_decoder_hidden_state = whisper_outputs.last_hidden_state
        # whisper_logits = whisper_outputs.logits[:, -1, :]
        whisper_logits = whisper_outputs.logits
        tcpgen_logits = self.tcpgen(original_logits=whisper_logits, prefix=decoder_input_ids, decoder_state=final_decoder_hidden_state)

        return (tcpgen_logits, final_decoder_hidden_state)

    def transcribe_with_biasing(self, raw_audio, sr=16000) -> str:
        # Preprocess audio using the feature extractor.
        self.model.eval()
        device = next(self.model.parameters()).device
        inputs = self.feature_extractor(raw_audio, sampling_rate=sr , return_tensors="pt").input_features
        inputs = inputs.to(device)
        
        # Create our custom logits processor with TCPGen and tokenizer.
        logits_processor = [TCPGenLogitsProcessor(self.tcpgen)]
        
        # Generate token IDs using the model while applying our biasing processor.
        generated_ids = self.model.generate(inputs, logits_processor=logits_processor)
        
        # Decode the generated token IDs to get the transcription.
        transcription = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription
