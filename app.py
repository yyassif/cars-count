import os
from datetime import date
from io import BytesIO

import boto3
import cv2
import torch
# from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, url_for
from PIL import Image

# load_dotenv()

BUCKET_NAME = os.getenv("AWS_BUCKET")
AWS_REGION = os.getenv("AWS_REGION")
FOLDER_NAME = "cars-count/images"
BASE_URL = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/"
S3 = boto3.client('s3')

app = Flask(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
@app.route("/index")
def index():
  return render_template("index.html")

@app.route('/about')
def about():
  return render_template('about.html')

@app.route('/prediction')
def prediction():
    result_image = request.args.get('result_image', None)
    result_number = request.args.get('result_number', None)
    return render_template('prediction.html', result_image=result_image, result_number=result_number)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
  if request.method == 'POST':
    # Check if the post request has the file part
    if 'image' not in request.files:
      return redirect(request.url)
    
    image_file = request.files['image']
    
    if image_file.filename == '':
      return redirect(request.url)
    
    if image_file and allowed_file(image_file.filename):

      # Read the image from the request
      image = Image.open(image_file.stream)
      uploaded_image_path = upload_image(image, image_file.filename.split(".")[-1], image_file.filename)

      if uploaded_image_path:
        model = torch.load('./model/yolov8.pt')
        results = model.predict([uploaded_image_path], save=True, max_det=8000 , conf=0.3 , save_txt=True, hide_labels=True, hide_conf=True)

        result_number = 0
        for result in results:
            boxes = result.boxes.cpu().numpy()
            counts += len(boxes)

        buffer = BytesIO() 
        result = results[0]
        orig_img = result.orig_img
        boxes = result.boxes

        for box in boxes:
            x_min, y_min, x_max, y_max, score, class_id = box.boxes.tolist()[0]
            cv2.rectangle(orig_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        # Convert the image array to a PIL Image
        PIL_image = Image.fromarray(orig_img_rgb)

        # Save the PIL Image to the in-memory buffer
        PIL_image.save(buffer, format=image_file.filename.split(".")[-1])
        uploaded_image_path = upload_image(PIL_image, image_file.filename.split(".")[-1], image_file.filename)
        return redirect(url_for('prediction', result_image=uploaded_image_path, result_number=result_number))

  return render_template('predict.html')

def upload_image(image, filetype, filename):
  try:
    filetype = "jpeg" if filetype == "jpg" else filetype
    date_now = date.today().strftime("%Y-%m-%d")
    SAVE_PATH = f"{FOLDER_NAME}/{date_now}-{filename}"
    buffer = BytesIO()
    image.save(buffer, format=filetype)
    S3.put_object(Body=buffer.getvalue(), Bucket=BUCKET_NAME, Key=SAVE_PATH)
    return BASE_URL + SAVE_PATH
  except Exception as e:
    print("Couldn't upload the image")
  return None

if __name__ == '__main__':
  app.run()

