from gevent.pywsgi import WSGIServer
import flask
import numpy as np
from flask import Flask,flash, request, jsonify, render_template, redirect,url_for
import cv2
import easyocr
from autocorrect import Speller
from PIL import Image
import numpy as np
from englisttohindi.englisttohindi import EngtoHindi 
import PIL
from PIL import Image, ImageDraw, ImageFont
from autocorrect import Speller
import re
from werkzeug.utils import secure_filename
import os



UPLOAD_FOLDER = 'New folder (2)/uploads/'

app = Flask(__name__, template_folder='template',static_folder=r'C:\Users\user\New folder (2)')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

font_hindi = ImageFont.truetype(r"C:\Users\user/Gargi.ttf",25)


def english_to_hindi(text):
  '''
  This function accepts the english sentence or word,
  and converts it to hindi language
  '''
  
  #creating a EngtoHindi() object 
  res = EngtoHindi(text) 
  
  #converting english text to hindi text
  converted_text = res.convert

  return converted_text

def easyocr_engine(image):
  
  coords = []
  image_copy = image.copy()
  image_copy1 = image.copy()
  reader = easyocr.Reader(['en'])  
  op = reader.readtext(image)
  pred_words=[]
  for i in range(len(op)):
    pts = np.array(op[i][0])
    coords.append(np.int32(pts))
    cv2.polylines(image, np.int32([pts]),True,(0,255,0),2)
    cv2.polylines(image_copy1, np.int32([pts]),True,(0,255,0),2)
    
    ## https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
    cv2.putText(image, op[i][1], (int(op[i][0][0][0]), int(op[i][0][0][1]-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
    pred_words.append(op[i][1])
  
  return image_copy,image_copy1, coords,pred_words

def gt(path):
  li = [];g_t=[]
  f = open(path, "r",encoding='utf-8-sig')
  for x in f:
    li.append(x.split())
  for i in li: ## Taking all the coordinates of text region in that image and appending that in a list boxes
    g_t.append(i[4].replace("\n",""))
  g_t = np.array(g_t)
  
  return g_t


def normalize_string(w):
  try:
    w = re.sub(r"[^a-zA-Z]+", ' ', w)
    return w.lower()
  except:
    pass

def detect_texts(our_image,trans,gt):
      
    ## text detection and recognition
    img,img_det,coords,pred_ = easyocr_engine(our_image) ## img ===> img_with_text; img_det ===> img with bbs for detected text
    
    detected_img_path_n_pts = [] 
    #Removing existing unneeded detetcted cropped image
    for index,pts in enumerate(coords):
      for i in range(len(pts)):
        for j in range(len(pts[i])):
          if (pts[i][j] < 0):
            pts[i][j] = 0
      rect = cv2.boundingRect(pts)
      x,y,w,h = rect
      croped = img[y:y+h, x:x+w].copy()
      pts = pts - pts.min(axis=0)
      mask = np.zeros(croped.shape[:2], np.uint8)
      cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
      dst = cv2.bitwise_and(croped, croped, mask=mask)
      #adding the white background
      bg = np.ones_like(croped, np.uint8)*255
      cv2.bitwise_not(bg,bg, mask=mask)
      dst2 = bg + dst
      img_path = "detected-text/"+str(i)+".png"
      #saving detected cropped text instance image
      cv2.imwrite(img_path, dst2)
      temp,n = (x,y),img_path
      detected_img_path_n_pts.append(temp)
    
    pred_words = normalize_string(" ".join(pred_).lower()).split()

    spell = Speller(lang='en')
    for i in range(len(pred_words)):
        pred_words[i] = spell(pred_words[i])

    if trans:
      #Translating predicted text to specified language
      pred_trans = []
      for i in pred_:
        if i.isdigit():
          pred_trans.append(i)
        else:
          pred_trans.append(english_to_hindi(i))

      img_pil = Image.fromarray(img_det) ##img_det is image with only bbs
      draw = ImageDraw.Draw(img_pil)

      for i in range(len(pred_trans)):
        draw.text(detected_img_path_n_pts[i], pred_trans[i], font=font_hindi,fill='black')
      img_det = np.array(img_pil)
    else:
      for i in range(len(pred_)):
        cv2.putText(img_det, pred_[i], detected_img_path_n_pts[i], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
    
    
    actual = normalize_string("|".join(list(gt)).lower()).split()
    #print(actual)
    y_true = 0
    for i in pred_words:
      for j in actual:
        if i == j:
          y_true +=1
    
    
    if trans:
      return img_det,actual,pred_words,y_true,len(actual)-y_true,(y_true*100)/(len(actual)),pred_trans
    return img_det,actual,pred_words,y_true,len(actual)-y_true,(y_true*100)/(len(actual)),[]

@app.route('/')
def home():
    return render_template('i.html')

@app.route('/',methods=['POST'])
def upload_image():
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #print('upload_image filename: ' + filename)
    flash('Image successfully uploaded and displayed below')
    return render_template('i.html', filename=filename)

@app.route('/display/<filename>')
def display_image(filename):
	#return 'display_image filename: ' + filename
	return filename

@app.route('/predict',methods=['POST'])
def predict():
    
    f = request.files['file']
    i = secure_filename(f.filename)
    image = cv2.imread(i)
    #cv2.imshow('',image)
    result_img,Actual_text_instances,Predicted_text_instances,correctly_recognized_word,incorrectly_recognized_word,Accuracy,pred_trans= detect_texts(image,trans=True,gt=gt(r"C:\Users\user/gt_100.txt"))

    #final_features = detect_texts(image_link[0],image_link[1])
    #prediction,score = beam_search(final_features,3)
    
    return render_template('i.html', prediction_text = 'Predicted_texts: {},\npred_trans:{}'.format(','.join(Predicted_text_instances),(','.join(pred_trans))))


    


if __name__ == '__main__':
    app.run(debug=True)