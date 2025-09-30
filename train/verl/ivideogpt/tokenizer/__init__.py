from .vq_model import CNNFSQModel256
from ivideogpt.ctx_tokenizer.compressive_vq_model import CompressiveVQModelFSQ

TOKENIZER = {
    "cnn": CNNFSQModel256,
    "ctx_cnn": CompressiveVQModelFSQ,
}