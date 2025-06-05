import os
from flask import Flask, redirect, render_template, request from PIL import Image
import torchvision.transforms.functional as TF import CNN
import numpy as np import torch
import pandas as pd import csv

supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')
model = CNN.CNN(39)
# model.load_state_dict(torch.load("plant_disease_model_1_latest.pt")) model.eval()

def recommend_cosmetics(skin_type): if skin_type == "Normal Skin":
return """For normal skin, you're lucky to have a well-balanced complexion. Your primary goal is to maintain your skin's health. Here's a comprehensive skincare routine:
1.Cleanser: Use a gentle, sulfate-free cleanser to remove impurities.
2.Moisturizer: Opt for a lightweight, non-comedogenic moisturizer.
3.Sunscreen: Apply a broad-spectrum sunscreen daily to protect your
skin.
4.Optional: You can incorporate a mild exfoliant 1-2 times a week for extra glow.
"""
elif skin_type == "Sensitive Skin":
return """Sensitive skin requires extra care to minimize irritation and redness. Consider these steps:
1.Cleanser: Use a fragrance-free, hypoallergenic cleanser.
2.Moisturizer: Choose a product with soothing ingredients like aloe vera or chamomile.
3.Sunscreen: Use a physical sunscreen with zinc oxide or titanium dioxide.
4.Avoid harsh exfoliants and strong active ingredients, and patch-test new products.
"""
elif skin_type == "Dry Skin":
return """Dry skin needs intense hydration and protection. Follow this regimen:
1.Cleanser: Use a hydrating, gentle cleanser.
2.Moisturizer: Opt for a rich, creamy moisturizer with ingredients like hyaluronic acid or ceramides.
3.Sunscreen: Apply a broad-spectrum sunscreen daily to prevent further dryness.
4.Consider adding a hydrating serum or facial oil to your routine for added moisture.
"""
elif skin_type == "Oily Skin":
return """To control excess oil and minimize breakouts, follow these
steps:
1.Cleanser: Use a foaming, salicylic acid-based cleanser to control
oil.
2.Moisturizer: Choose an oil-free, lightweight, and non-comedogenic moisturizer.
3.Sunscreen: Use an oil-free, mattifying sunscreen.
4.Consider using products with ingredients like salicylic acid, niacinamide, or witch hazel to manage oil and acne.
"""
elif skin_type == "Scaly Skin":
return """Scaly skin often results from dryness and flakiness. Try these skincare steps:
1.Cleanser: Use a gentle exfoliating cleanser to remove dead skin
cells.
2.Moisturizer: Choose a rich, emollient moisturizer to lock in moisture.
3.Sunscreen: Protect your skin from further damage with a daily sunscreen.
4.Exfoliate with products containing glycolic acid or lactic acid to improve texture.
"""
elif skin_type == "Red_Spots_skin":
return """Red spots can be due to various causes. Here's a general approach:
1.Cleanser: Use a gentle, fragrance-free cleanser to avoid irritation.
2.Moisturizer: Select a calming and hydrating moisturizer.
3.Sunscreen: Shield your skin from further damage with a broad- spectrum sunscreen.
4.Consult a dermatologist to identify the specific cause of redness and receive tailored treatment.
"""
elif skin_type == "Skin_moles":
return """Moles are usually harmless but require care. Follow these guidelines:
1. Sunscreen: Protect your skin with a broad-spectrum sunscreen to
prevent sun damage.
2.Regularly examine your moles for any changes in size, shape, or
color.
3.If you notice changes in a mole, consult a dermatologist for a thorough evaluation.
4.Avoid sun exposure, and consider wearing protective clothing and
hats.
"""
else:
return "Please enter a valid skin type."
def prediction(image_path):
import tensorflow as tf
model = tf.keras.models.load_model('skin/skin/model.h5') import numpy as np
import matplotlib.pyplot as plt import warnings warnings.filterwarnings(action='once')
from tensorflow.keras.preprocessing import image def prepare(img_path):
img = image.load_img(img_path, target_size=(224,224)) x = image.img_to_array(img)
x = x/255
return np.expand_dims(x, axis=0)
img_path = image_path
predictions = model.predict([prepare(img_path)])
skin_types =['Red_Spots_skin', 'Dry Skin', 'Normal Skin', 'Oily Skin', 'Scaly Skin', 'Sensitive Skin', 'Skin_moles']
predicted_skin_type = skin_types[np.argmax(predictions)]
# print(f'Predicted Skin Type: {predicted_skin_type}')
# # Generate skincare recommendations based on the predicted skin type # recommendations = recommend_cosmetics(predicted_skin_type)
# print('Skincare Recommendations:')
# print(recommendations)
return predicted_skin_type
app = Flask( name )
@app.route('/') def home_page():
return render_template('home.html')

@app.route('/contact') def contact():
return render_template('contact-us.html')
@app.route('/index')
def ai_engine_page():
return render_template('index.html')
@app.route('/mobile-device')
def mobile_device_detected_page():
return render_template('mobile-device.html')
@app.route('/submit222', methods=['POST']) def submit222():
text = request.form['textfield']
with open('data.csv', 'a', newline='') as csvfile: fieldnames = ['text']

writer = csv.DictWriter(csvfile, fieldnames=fieldnames) writer.writerow({'text': text})
return 'Review submitted successfully!'
@app.route('/submit', methods=['GET', 'POST']) def submit():
if request.method == 'POST':
image = request.files['image'] filename = image.filename
file_path = os.path.join('static/uploads', filename) image.save(file_path)
print(file_path)
pred = prediction(file_path)
# title = disease_info['disease_name'][pred]
# description =disease_info['description'][pred] # prevent = disease_info['Possible Steps'][pred] # image_url = disease_info['image_url'][pred]
# supplement_name = supplement_info['supplement name'][pred]
# supplement_image_url = supplement_info['supplement image'][pred] # supplement_buy_link = supplement_info['buy link'][pred]
# return render_template('submit.html' , title = title , desc = description , prevent = prevent ,
#	image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

# # Generate skincare recommendations based on the predicted skin type recommendations = recommend_cosmetics(pred)
print('Skincare Recommendations:') print(recommendations)


return render_template('submit.html' , pred =
pred,recommendations=recommendations)

 @app.route('/market', methods=['GET', 'POST']) def market():
return render_template('market.html', supplement_image =
list(supplement_info['supplement image']),
supplement_name = list(supplement_info['supplement
name']),disease_name= list(supplement_info['disease_name']), buy = list(supplement_info['buy link']))

 
if  name  == ' main ':
app.run(debug=True)



