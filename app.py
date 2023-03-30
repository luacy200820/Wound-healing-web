from flask import Flask,render_template,request,jsonify,send_from_directory
from werkzeug.utils import secure_filename
from datetime import timedelta
from tensorflow.keras.utils import CustomObjectScope
import segmentation_models as sm
import tensorflow as tf 
import os 
import cv2
import numpy as np 
import time 
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.dirname(__file__)


@app.route('/')
def test():
    return "test web"

ALLOWED_EXTENSIONS = set(['png','jpg','JPG','PNG'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

app.send_file_max_age_default = timedelta(seconds=1)

@app.route('/home',methods=['POST','GET'])
def upload():
    if request.method == "POST":
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"effor":1001,"msg":"圖片類型: png, jpg, PNG, JPG"})
        filename = secure_filename(f.filename)
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"],'static\\images',filename)
        f.save(upload_path)
        print("filename",upload_path,filename)
    
        result = predict(upload_path,filename)
        filename_result = result.split('\\')[-1]
        # filename = secure_filename(result)
        print("result filename",filename)
        return render_template('home.html',outputImageName = filename_result,imagename = filename)
    return render_template('home.html')

def result_mask(result,classes=4):
    # w,h = result.shape[:2]
    result_rgb = np.zeros((256,256,3),dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            max_value = max(result[i,j,0], result[i,j,1],result[i,j,2],result[i,j,3])
            # max_value = max(result[i,j,0], result[i,j,1])
            if max_value == result[i,j,0]:
                result_rgb[i,j,0] = 0
                result_rgb[i,j,1] = 0
                result_rgb[i,j,2] = 0
            elif max_value == result[i,j,1]:
                result_rgb[i,j,0] = 1
                result_rgb[i,j,1] = 0
                result_rgb[i,j,2] = 0
            elif max_value == result[i,j,2]:
                result_rgb[i,j,0] = 0
                result_rgb[i,j,1] = 1
                result_rgb[i,j,2] = 1
            elif max_value == result[i,j,3]:
                result_rgb[i,j,0] = 0
                result_rgb[i,j,1] = 0
                result_rgb[i,j,2] = 1 
    return result_rgb

def predict(filename,partName):
    print("predict")
    # name = filename.split('\\')[-1]
    name = partName
    start = time.time()
    img = cv2.imread(filename)
    img_ = img
    img = cv2.resize(img,(256,256))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    img = img /255.
    with CustomObjectScope({ 'f1-score': sm.metrics.f1_score,'iou_score':sm.metrics.iou_score}):
        model = tf.keras.models.load_model('deeplabv3CIENO.h5')
    
    pred = model.predict(img.reshape(1,256,256,3))[0]
    output_image = result_mask(pred)
    end = time.time()
    print("time spend",str(end-start))
    # upload_path = os.path.join(app.config["UPLOAD_FOLDER"],'static\\output',output_image)
    dst = app.config["UPLOAD_FOLDER"]+'\\static\\output\\'+name
    mix_img = cv2.addWeighted(img_,0.7,output_image*255,0.3,5)

    cv2.imwrite(dst,mix_img)
    # cv2.imshow('output',output_image*255)
    # cv2.imshow('mix',mix_img)
    # cv2.waitKey(0)
    print(dst)
    print("finish")
    # return render_template('home.html',outputImageName = dst)
    return dst
    
    
    
# @app.route('/upload/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config["UPLOAD_FOLDER"],filename)
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug = True )
