from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import sys
import subprocess
from werkzeug.utils import secure_filename
import shutil
import yaml
from yaml import Loader
import logging

app = Flask(__name__)
CORS(app, resources={
    r"/split": {"origins": "http://localhost:3000"},
    r"/outputs/*": {"origins": "http://localhost:3000"}
})

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_file_exists(filepath):
    if os.path.exists(filepath):
        logger.info(f"File exists: {filepath}")
        if os.access(filepath, os.R_OK):
            logger.info(f"File is readable: {filepath}")
        else:
            logger.warning(f"File is not readable: {filepath}")
    else:
        logger.error(f"File does not exist: {filepath}")

@app.route('/outputs/<path:path>', methods=['GET'])
def serve_output_files(path):
    return send_from_directory(app.config['OUTPUT_FOLDER'], path)

@app.route('/split', methods=['POST'])
def split_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Clear the uploads folder
        uploads_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
        for file_path in os.listdir(uploads_path):
            file_path = os.path.join(uploads_path, file_path)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        
        logger.info(f"Cleared uploads folder: {uploads_path}")
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            output_dir = os.path.join(app.config['OUTPUT_FOLDER'], os.path.splitext(filename)[0])
            os.makedirs(output_dir, exist_ok=True)

            # Check if files exist and are readable
            check_file_exists('configs/config_vocals_mel_band_roformer.yaml')
            check_file_exists('MelBandRoformer.ckpt')
            check_file_exists('inference.py')
            check_file_exists(file_path)

            # Load configuration from YAML file
            try:
                with open('configs/config_vocals_mel_band_roformer.yaml') as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
                logger.info("Successfully loaded config file")
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
                raise

            # Construct paths for logging
            uploads_path = os.path.abspath(app.config['UPLOAD_FOLDER'])
            outputs_path = os.path.abspath(output_dir)
            config_path = os.path.abspath('configs/config_vocals_mel_band_roformer.yaml')
            model_path = os.path.abspath('MelBandRoformer.ckpt')

            # Run inference.py script to process the audio file
            cmd = [
                sys.executable,
                "inference.py",
                "--config_path", config_path,
                "--model_path", model_path,
                "--input_folder", uploads_path,
                "--store_dir", outputs_path
            ]
            
            logger.info(f"Uploaded file: {filename}")
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Uploads directory: {uploads_path}")
            logger.info(f"Outputs directory: {outputs_path}")
            logger.info(f"Config path: {config_path}")
            logger.info(f"Model path: {model_path}")
            logger.info(f"Executing command: {' '.join(cmd)}")
            
            # Run command and capture output
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = proc.communicate()
            
            # Print captured output
            print(stdout)
            if stderr:
                print(stderr, file=sys.stderr)
            
            if proc.returncode == 0:
                vocals_filename = f"{os.path.splitext(filename)[0]}_{config['training']['target_instrument']}.wav"
                instrumental_filename = f"{os.path.splitext(filename)[0]}_instrumental.wav"
                
                output_dir_name = os.path.basename(output_dir)
                vocals_path = os.path.join(OUTPUT_FOLDER, output_dir_name, vocals_filename).replace('\\', '/')
                instrumental_path = os.path.join(OUTPUT_FOLDER, output_dir_name, instrumental_filename).replace('\\', '/')
                
                vocals_size = os.path.getsize(os.path.join(output_dir, vocals_filename))
                instrumental_size = os.path.getsize(os.path.join(output_dir, instrumental_filename))
                
                logger.info(f"Vocals file size: {vocals_size} bytes")
                logger.info(f"Instrumental file size: {instrumental_size} bytes")
                
                return jsonify(
                    {
                        'vocals': vocals_path,
                        'accompaniment': instrumental_path
                    }
                ), 200
            else:
                logger.error(f"Error processing file. Exit code: {proc.returncode}")
                return jsonify({'error': f'An error occurred while processing the file'}), 500
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return jsonify({'error': f'An error occurred while processing the file'}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    logger.info("Flask app is ready to receive requests.")
    app.run(debug=True, host="0.0.0.0")