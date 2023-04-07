from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import cv2
import sys
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
app=Flask(__name__)
app.config['UPLOAD_FOLDER']='static/uploads/'
@app.route('/')
def upload_form():
    return render_template('upload_form.html')
@app.route('/', methods=['POST'])
def upload_image():
    import os
    from PIL import Image, ImageChops, ImageEnhance
    import os
    import itertools
    import numpy as np
    file = request.files['image']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    image = Image.open(filepath)
    image.save(filepath)
    image=np.uint8(image)
    # print(image)
    from tensorflow.keras.models import load_model
    import numpy as np
    model = load_model("C:/Users/Sandeep/Desktop/main project/model/image_splicing.h5")
    # print(model)
    

    def convert_to_ela_image(path, quality):
        
        temp_filename = 'temp_file.jpg'
        filename2 = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        ela_filename = 'temp_ela_file.png'
    
        image = Image.open(path).convert('RGB')
        image.save(filename2, 'JPEG', quality = quality)
        temp_image = Image.open(temp_filename)
    
        ela_image = ImageChops.difference(image, temp_image)
    
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
    
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
        return ela_image
    # real_image_path = filepath
    # Image.open(real_image_path)
    image_size = (128, 128)

    def prepare_image(image_path):
        return np.array(convert_to_ela_image(image_path, 85).resize(image_size)).flatten() / 255.0
    # convert_to_ela_image(real_image_path, 85)
    # import numpy as np
    path = filepath
    x2 = prepare_image(path)
    x2 = x2.reshape(-1, 128, 128, 3)
    arr = model.predict(x2)
    print(arr)
    count=0
    if(arr[0][0]>arr[0][1]):
        count=count+1
        print("IMAGE IS TAMPERED")
    else:
        count=count+2
        print("IMAGE IS AUTHENTICATED")
    filename2 ="C:/Users/Sandeep/Desktop/projects/image-splicing-detection/static/uploads/temp_file.jpg"
    filename1='temp_file.jpg'
    filepath3 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    image = Image.open(filepath3)
    image=np.uint8(image)
    print(image)
    if count==2:
       position=(10,50)
       print(count)
       cv2.putText(image, "AUTHENTICATED",position, cv2.FONT_HERSHEY_SIMPLEX, 1, (211, 80, 0, 255), 3) 
       cv2.imwrite('sample_out_2.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    #    cv2.imshow('img',image)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
       filename1='savedimage.jpg'
    
       filename2 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    #    image.save(filename2)
    #    print(filename2)
       cv2.imwrite(filename2,cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    #    print(image)
    if count==1:
       position=(10,50)
       print(count)
       cv2.putText(image, "TAMPERED",position, cv2.FONT_HERSHEY_SIMPLEX, 1, (211, 80, 0, 255), 3) 
       cv2.imwrite('sample_out_2.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    #    cv2.imshow('img',image)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
       filename1='savedimage.jpg'
    
       filename2 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    #    image.save(filename2)
    #    print(filename2)
       cv2.imwrite(filename2,cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # if count==2:
    #     position=(10,50)
    #     cv2.putText(
    #  image, #numpy array on which text is written
    #  "No Forgery Found!", #text
    #  position, #position at which writing has to start
    #  cv2.FONT_HERSHEY_SIMPLEX, #font family
    #  1, #font size
    #  (209, 80, 0, 255), #font color
    #  3) #font s
        
    
    return redirect(url_for('view_image', filename=filename1))
@app.route('/view/<filename>')
def view_image(filename):
    print('hi')
    return render_template('view_image.html',filename=filename)
if __name__ == '__main__':
    app.run(debug=True)