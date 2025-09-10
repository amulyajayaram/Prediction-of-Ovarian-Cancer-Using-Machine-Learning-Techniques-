import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing import image
import cv2
import numpy as np
from tensorflow.keras.models import load_model
##from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request
import sqlite3
from keras.models import load_model
import shutil



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/userlog.html')
def userlogg():
    return render_template('userlog.html')

@app.route('/developer.html')
def developer():
    return render_template('developer.html')

@app.route('/graph.html', methods=['GET', 'POST'])
def graph():
    
    images = ['http://127.0.0.1:5000/static/accuracy_plot.jpg',
              'http://127.0.0.1:5000/static/loss_plot.png',
              'http://127.0.0.1:5000/static/confussion_mtrix.jpg']
    content=['Accuracy Graph',
             'Loss Graph(Error Message)',
            'Confusion Matrix']

            
    
        
    return render_template('graph.html',images=images,content=content)
    


@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
 
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        

        shutil.copy("test/"+fileName, dst)
        image = cv2.imread("test/"+fileName)
        #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        #apply the Canny edge detection
        edges = cv2.Canny(image, 250, 254)
        cv2.imwrite('static/edges.jpg', edges)
        #apply thresholding to segment the image
        retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Thresholding
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Cell counting
        total_cells = len(contours)

        # Damaged cell detection
        damaged_cells = 0
        min_area_threshold = 100  # Adjust this threshold based on your specific case

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area_threshold:
                damaged_cells += 1

        # Overlap detection
        overlap_cells = 0

        for i in range(len(contours)):
            for j in range(i + 1, len(contours)):
                # Calculate centroids
                M_i = cv2.moments(contours[i])
                M_j = cv2.moments(contours[j])

                # Ensure non-zero area before calculating centroids
                if M_i['m00'] != 0 and M_j['m00'] != 0:
                    cx_i = int(M_i['m10'] / M_i['m00'])
                    cy_i = int(M_i['m01'] / M_i['m00'])

                    cx_j = int(M_j['m10'] / M_j['m00'])
                    cy_j = int(M_j['m01'] / M_j['m00'])

                    # Calculate distance between centroids
                    distance = np.sqrt((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2)

                    if distance < 20:  # Adjust this threshold based on your specific case
                        overlap_cells += 1
        


       
        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        status1=''
        model=load_model('Convolutional_Neural_Network.h5')
        path='static/images/'+fileName
        # Load the trained CNN model
        cnn_model_path = "Convolutional_Neural_Network.h5"
        cnn_model = load_model(cnn_model_path)

        # Define the classes
        class_labels = os.listdir("D:\\2022project\\OVERIAN_CANCER\\test")

        # Function to prepare the image for prediction
        def prepare_test_image(path):
            img = load_img(path, target_size=(128, 128), grayscale=True)
            x = img_to_array(img)
            x = x / 255.0
            return np.expand_dims(x, axis=0)

        # Function to predict and display the result
##        def predict_and_display_image(model, img_path, class_labels):
        result = model.predict(prepare_test_image(path))
        img = cv2.imread(path)

        print("Prediction Result:", result[0])

        class_result = np.argmax(result, axis=1)
        print("Class number:", class_result[0])
        print("Predicted class:", class_labels[class_result[0]])


       
        result = list(result[0])
        if class_result[0] == 0:
            str_label = "CC(Clear-Cell Ovarian Carcinoma)"

            print("The predicted image of the CC is with a accuracy of {} %".format(result[class_result[0]]*100))
            accuracy="The predicted image of the cc is with a accuracy of {}%".format(result[class_result[0]]*100)
           
           
            
        elif class_result[0] == 1:
            str_label  = "EC(Endometrioid"
            status1="stage2 cancer"
            print("The predicted image of the EC(Endometrioid) is with a accuracy of {} %".format(result[class_result[0]]*100))
            accuracy="The predicted image of the EC(Endometrioid) is with a accuracy of {}%".format(result[class_result[0]]*100)
           
            

        elif class_result[0] == 2:
            str_label  = "HGSC(High-Grade Serous Carcinoma)"
            
            print("The predicted image of the HGSC is with a accuracy of {} %".format(result[class_result[0]]*100))
            accuracy="The predicted image of the HGSC is with a accuracy of {}%".format(result[class_result[0]]*100)
            
           

        elif class_result[0] == 3:
            str_label  = "LGSC(Low-Grade Serous)"
            print("The predicted image of the LGSC(Low-Grade Serous) is with a accuracy of {} %".format(result[class_result[0]]*100))
            accuracy="The predicted image of the LGSC(Low-Grade Serous) is with a accuracy of {}%".format(result[class_result[0]]*100)





        elif class_result[0] == 4:
            str_label  = "MC(Mucinous Carcinoma)"
            print("The predicted image of the MC(Mucinous Carcinoma) is with a accuracy of {} %".format(result[class_result[0]]*100))
            accuracy="The predicted image of the MC(Mucinous Carcinoma) is with a accuracy of {}%".format(result[class_result[0]]*100)
        A=float(result[0])
        B=float(result[1])
        C=float(result[2])
        D=float(result[3])
        E=float(result[4])
        
        
        dic={'CC':A,'EC':B,'HGSC':C,'LGSC':D,'MC':E}
        algm = list(dic.keys()) 
        accu = list(dic.values()) 
        fig = plt.figure(figsize = (5, 5))  
        plt.bar(algm, accu, color ='maroon', width = 0.3)  
        plt.xlabel("Comparision") 
        plt.ylabel("Accuracy Level") 
        plt.title("Accuracy Comparision between \n Ovarian Cancer Detection")
        plt.savefig('static/matrix.png')


        # Print results
        print("Total Cells: {}".format(total_cells))
        print("Damaged Cells: {}".format(damaged_cells))
        print("Overlap Cells: {}".format(overlap_cells))
        cell_count = [total_cells, damaged_cells, overlap_cells]

        
            

        return render_template('userlog.html', status=str_label,Label=status1,accuracy=accuracy,cell_count=cell_count,ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",ImageDisplay4="http://127.0.0.1:5000/static/matrix.png")
        
    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
