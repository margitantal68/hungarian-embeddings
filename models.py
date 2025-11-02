import ollama
import openai
import os 
import torch
import numpy as np

from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel 
from google import genai
from google.genai import types

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


class SentenceTransformerEmbedder:
    def __init__(self, model_name, normalize_embeddings=True):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.normalize_embeddings = normalize_embeddings
        # print(self.model)
        # print("-----------------------------------")


    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=None):
        if isinstance(texts, str):
            texts = [texts]

        normalize = self.normalize_embeddings if normalize_embeddings is None else normalize_embeddings
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize
        )
        return embeddings
    

class OllamaEmbedder:
    def __init__(self, model_name="nomic_embed_text"):
        self.model_name = model_name
        self.ollama = ollama

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            response = self.ollama.embeddings(model=self.model_name, prompt=text)
            emb = response["embedding"]
            embeddings.append(emb)
        arr = np.array(embeddings, dtype=np.float32)
        if normalize_embeddings:
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        return arr

class InstructorEmbedder:
    def __init__(self, model_name="hkunlp/instructor-large", normalize_embeddings=True):
        self.model = INSTRUCTOR(model_name)
        self.normalize_embeddings = normalize_embeddings
        # Access tokenizer's model_max_length
        # tokenizer_max = self.model.tokenizer.model_max_length
        # print(f"Model name: {model_name}    Tokenizer max length: {tokenizer_max}")


    def encode(self, texts, instruction="Represent the text:", convert_to_numpy=True, normalize_embeddings=None):
        """
        Encode text(s) with an instruction.

        Args:
            texts (str or list[str]): Text or list of texts to embed.
            instruction (str): Instruction describing how to represent the text(s).
            convert_to_numpy (bool): Whether to convert embeddings to numpy arrays.
            normalize_embeddings (bool): Whether to normalize embeddings (overrides class default).

        Returns:
            np.ndarray or torch.Tensor: The resulting embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Combine instruction with each text as required by INSTRUCTOR
        instruction_text_pairs = [[instruction, text] for text in texts]

        normalize = self.normalize_embeddings if normalize_embeddings is None else normalize_embeddings

        embeddings = self.model.encode(
            instruction_text_pairs,
            normalize_embeddings=normalize,
            convert_to_numpy=convert_to_numpy
        )
        return embeddings


class OpenAIEmbedder:
    def __init__(self, model_name="text-embedding-ada-002"):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        
        embeddings = [e.embedding for e in response.data]
        arr = np.array(embeddings, dtype=np.float32)

        if normalize_embeddings:
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)

        return arr


# Adapted for HuRTE - max 100 texts per embed_content call
# This is a workaround for the Gemini API limit of 100 texts per call.
# If you have more than 100 texts, you should split them into chunks.


def batch_texts(texts, batch_size=100):
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]

class GeminiEmbedder:
    # def __init__(self, model_name="models/embedding-001"):
    #     self.model_name = model_name

    def __init__(self, model_name="gemini-embedding-001"):
        self.model_name = model_name

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]

        client = genai.Client(api_key=GEMINI_API_KEY)
        arr = []
        model ='gemini-embedding-001'
        if len(texts) <= 100:
            response = client.models.embed_content(
            model=model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=768))
            arr = np.vstack([emb.values for emb in response.embeddings])
        else:
            for batch in batch_texts(texts, batch_size=100):
                response = client.models.embed_content(
                    model=model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=768,
                    )
                )
                arr.extend([emb.values for emb in response.embeddings])
            arr = np.vstack(arr)
        
        if normalize_embeddings:
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)

        return arr




"""
A simple wrapper class for generating Hungarian text embeddings
using the 'SZTAKI-HLT/hubert-base-cc' model (or any compatible Hugging Face model).
"""


class HubertEmbedder:
    def __init__(self, model_name="SZTAKI-HLT/hubert-base-cc", normalize_embeddings=True, device=None):
        """
        Initialize the HuBERT model and tokenizer.

        Args:
            model_name (str): Hugging Face model identifier.
            normalize_embeddings (bool): Whether to normalize embeddings to unit length.
            device (str): 'cuda' or 'cpu'. Defaults to CUDA if available.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name, torch_dtype="auto").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.normalize_embeddings = normalize_embeddings
   
        # print(f"Model name: {model_name}    Model: {self.model}")
        # print("-----------------------------------")

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=None):
        """
        Generate sentence embeddings by averaging token embeddings.

        Args:
            texts (str or list[str]): Input text(s).
            convert_to_numpy (bool): Return embeddings as NumPy arrays.
            normalize_embeddings (bool): Override normalization flag.

        Returns:
            np.ndarray or torch.Tensor: Sentence embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        normalize = self.normalize_embeddings if normalize_embeddings is None else normalize_embeddings

        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Get hidden states
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # Compute mean pooling
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        masked_embeddings = hidden_states * attention_mask
        sum_embeddings = masked_embeddings.sum(dim=1)
        valid_token_counts = attention_mask.sum(dim=1)
        sentence_embeddings = sum_embeddings / valid_token_counts

        if normalize:
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        if convert_to_numpy:
            # sentence_embeddings = sentence_embeddings.cpu().numpy()
            sentence_embeddings = sentence_embeddings.cpu().numpy().astype(np.float32)

        return sentence_embeddings


