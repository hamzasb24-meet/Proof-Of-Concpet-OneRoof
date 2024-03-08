
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import accelerate
def ASR(sample):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "distil-whisper/distil-small.en"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )
    result = pipe(sample)
    return result["text"]




from flask import Flask, render_template, request

import os
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'super-secret-key'




UPLOAD_FOLDER = 'static/Upload_Folder'
ALLOWED_EXTENSIONS = {'mp3'}
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'super-secret-key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/' , methods=['GET' , 'POST'])
def home():
    return render_template("index.html")

@app.route('/result' , methods=['GET' , 'POST'])
def upload():
    if request.method == 'POST':
        # try:
            if 'file' not in request.files:
                return render_template("result.html", error=True, text="")
            file = request.files['file']
            if file.filename == '':
                return render_template("result.html", error="")
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                path = 'static/Upload_Folder/' + filename
                text = ASR(path)
                return render_template("result.html" , error=False , text=text)

        # except:
        #     return render_template("result.html", error=True , text="")


if __name__ == '__main__':
    app.run(debug=True)