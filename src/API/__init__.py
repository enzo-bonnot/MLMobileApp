import string
from flask import Flask
app = Flask(__name__)
from src.API.utils.network.model import HTRModel
from src.API.utils.generator import Tokenizer

app.tokenizer = Tokenizer(chars=string.printable[:95], max_text_length=128)
app.model = HTRModel(architecture='flor',
                     input_size=(1024, 128, 1),
                     vocab_size=app.tokenizer.vocab_size,
                     top_paths=10)

app.model.compile()
app.model.load_checkpoint(target='./checkpoint_recognizer.hdf5')

from src.API import routes