# save this as app.py
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, Markup
from markupsafe import escape
import pickle

app = Flask(__name__)   


fertilizer_dic = {
    
    'No': """All the macronutrients of your soil are normal.
        """,

        'NHigh': """The N value of soil is high and might give rise to weeds.
        <br/> Please consider the following suggestions:

        <br/><br/> 1. <i> Manure </i> – adding manure is one of the simplest ways to amend your soil with nitrogen. Be careful as there are various types of manures with varying degrees of nitrogen.

        <br/> 2. <i>Coffee grinds </i> – use your morning addiction to feed your gardening habit! Coffee grinds are considered a green compost material which is rich in nitrogen. Once the grounds break down, your soil will be fed with delicious, delicious nitrogen. An added benefit to including coffee grounds to your soil is while it will compost, it will also help provide increased drainage to your soil.

        <br/>3. <i>Plant nitrogen fixing plants</i> – planting vegetables that are in Fabaceae family like peas, beans and soybeans have the ability to increase nitrogen in your soil

        <br/>4. Plant ‘green manure’ crops like cabbage, corn and brocolli

        <br/>5. <i>Use mulch (wet grass) while growing crops</i> - Mulch can also include sawdust and scrap soft woods""",

        'Nlow': """The N value of your soil is low.
        <br/> Please consider the following suggestions:
        <br/><br/> 1. <i>Add sawdust or fine woodchips to your soil</i> – the carbon in the sawdust/woodchips love nitrogen and will help absorb and soak up and excess nitrogen.

        <br/>2. <i>Plant heavy nitrogen feeding plants</i> – tomatoes, corn, broccoli, cabbage and spinach are examples of plants that thrive off nitrogen and will suck the nitrogen dry.

        <br/>3. <i>Water</i> – soaking your soil with water will help leach the nitrogen deeper into your soil, effectively leaving less for your plants to use.

        <br/>4. <i>Sugar</i> – In limited studies, it was shown that adding sugar to your soil can help potentially reduce the amount of nitrogen is your soil. Sugar is partially composed of carbon, an element which attracts and soaks up the nitrogen in the soil. This is similar concept to adding sawdust/woodchips which are high in carbon content.

        <br/>5. Add composted manure to the soil.

        <br/>6. Plant Nitrogen fixing plants like peas or beans.

        <br/>7. <i>Use NPK fertilizers with high N value.

        <br/>8. <i>Do nothing</i> – It may seem counter-intuitive, but if you already have plants that are producing lots of foliage, it may be best to let them continue to absorb all the nitrogen to amend the soil for your next crops.""",

        'PHigh': """The P value of your soil is high.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Avoid adding manure</i> – manure contains many key nutrients for your soil but typically including high levels of phosphorous. Limiting the addition of manure will help reduce phosphorus being added.

        <br/>2. <i>Use only phosphorus-free fertilizer</i> – if you can limit the amount of phosphorous added to your soil, you can let the plants use the existing phosphorus while still providing other key nutrients such as Nitrogen and Potassium. Find a fertilizer with numbers such as 10-0-10, where the zero represents no phosphorous.

        <br/>3. <i>Water your soil</i> – soaking your soil liberally will aid in driving phosphorous out of the soil. This is recommended as a last ditch effort.

        <br/>4. Plant nitrogen fixing vegetables to increase nitrogen without increasing phosphorous (like beans and peas).

        <br/>5. Use crop rotations to decrease high phosphorous levels""",

        'Plow': """The P value of your soil is low.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Bone meal</i> – a fast acting source that is made from ground animal bones which is rich in phosphorous.

        <br/>2. <i>Rock phosphate</i> – a slower acting source where the soil needs to convert the rock phosphate into phosphorous that the plants can use.

        <br/>3. <i>Phosphorus Fertilizers</i> – applying a fertilizer with a high phosphorous content in the NPK ratio (example: 10-20-10, 20 being phosphorous percentage).

        <br/>4. <i>Organic compost</i> – adding quality organic compost to your soil will help increase phosphorous content.

        <br/>5. <i>Manure</i> – as with compost, manure can be an excellent source of phosphorous for your plants.

        <br/>6. <i>Clay soil</i> – introducing clay particles into your soil can help retain & fix phosphorus deficiencies.

        <br/>7. <i>Ensure proper soil pH</i> – having a pH in the 6.0 to 7.0 range has been scientifically proven to have the optimal phosphorus uptake in plants.

        <br/>8. If soil pH is low, add lime or potassium carbonate to the soil as fertilizers. Pure calcium carbonate is very effective in increasing the pH value of the soil.

        <br/>9. If pH is high, addition of appreciable amount of organic matter will help acidify the soil. Application of acidifying fertilizers, such as ammonium sulfate, can help lower soil pH""",

        'KHigh': """The K value of your soil is high</b>.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Loosen the soil</i> deeply with a shovel, and water thoroughly to dissolve water-soluble potassium. Allow the soil to fully dry, and repeat digging and watering the soil two or three more times.

        <br/>2. <i>Sift through the soil</i>, and remove as many rocks as possible, using a soil sifter. Minerals occurring in rocks such as mica and feldspar slowly release potassium into the soil slowly through weathering.

        <br/>3. Stop applying potassium-rich commercial fertilizer. Apply only commercial fertilizer that has a '0' in the final number field. Commercial fertilizers use a three number system for measuring levels of nitrogen, phosphorous and potassium. The last number stands for potassium. Another option is to stop using commercial fertilizers all together and to begin using only organic matter to enrich the soil.

        <br/>4. Mix crushed eggshells, crushed seashells, wood ash or soft rock phosphate to the soil to add calcium. Mix in up to 10 percent of organic compost to help amend and balance the soil.

        <br/>5. Use NPK fertilizers with low K levels and organic fertilizers since they have low NPK values.

        <br/>6. Grow a cover crop of legumes that will fix nitrogen in the soil. This practice will meet the soil’s needs for nitrogen without increasing phosphorus or potassium.
        """,

        'Klow': """The K value of your soil is low.
        <br/>Please consider the following suggestions:

        <br/><br/>1. Mix in muricate of potash or sulphate of potash
        <br/>2. Try kelp meal or seaweed
        <br/>3. Try Sul-Po-Mag
        <br/>4. Bury banana peels an inch below the soils surface
        <br/>5. Use Potash fertilizers since they contain high values potassium
        """
    }



# Loading fertilizer recommendation model

fert_recommendation_model_path = 'models/rf_pipeline.pkl'
fert_recommendation_model = pickle.load(open(fert_recommendation_model_path, 'rb'))

crop_dict = {0: 'Barley', 1: 'Cotton', 2: 'Ground Nuts', 3: 'Maize', 4: 'Millets', 5: 'Oil seeds', 6: 'Paddy', 7: 'Pulses', 8: 'Sugarcane', 9: 'Tobacco', 10: 'Wheat'}
soil_dict = {0: 'Black', 1: 'Clayey', 2: 'Loamy', 3: 'Red', 4: 'Sandy'}
ferti_dict = {0: '10-26-26', 1: '14-35-14', 2: '17-17-17', 3: '20-20', 4: '28-28', 5: 'DAP', 6: 'Urea'}

# Loading crop yield prediction model

# crop_yield_model_path = 'models/cropyielaptsmodel.pkl'
# crop_yield_model = pickle.load(open(crop_yield_model_path, 'rb'))

# state_dict = {0: 'Andhra Pradesh', 1: 'Telangana '}
# dist_dict = {0: 'ADILABAD', 1: 'ANANTAPUR', 2: 'CHITTOOR', 3: 'EAST GODAVARI', 4: 'GUNTUR', 5: 'HYDERABAD', 6: 'KADAPA', 7: 'KARIMNAGAR', 8: 'KHAMMAM', 9: 'KRISHNA', 10: 'KURNOOL', 11: 'MAHBUBNAGAR', 12: 'MEDAK', 13: 'NALGONDA', 14: 'NIZAMABAD', 15: 'PRAKASAM', 16: 'RANGAREDDI', 17: 'SPSR NELLORE', 18: 'SRIKAKULAM', 19: 'VISAKHAPATANAM', 20: 'VIZIANAGARAM', 21: 'WARANGAL', 22: 'WEST GODAVARI'}
# season_dict = {0: 'Kharif     ', 1: 'Rabi       ', 2: 'Whole Year '}
# cropname_dict = {0: 'Arecanut', 1: 'Arhar/Tur', 2: 'Bajra', 3: 'Banana', 4: 'Beans & Mutter(Vegetable)', 5: 'Bhindi', 6: 'Bottle Gourd', 7: 'Brinjal', 8: 'Cabbage', 9: 'Cashewnut', 10: 'Castor seed', 11: 'Citrus Fruit', 12: 'Coconut ', 13: 'Coriander', 14: 'Cotton(lint)', 15: 'Cowpea(Lobia)', 16: 'Cucumber', 17: 'Dry chillies', 18: 'Dry ginger', 19: 'Garlic', 20: 'Ginger', 21: 'Gram', 22: 'Grapes', 23: 'Groundnut', 24: 'Horse-gram', 25: 'Jowar', 26: 'Korra', 27: 'Lemon', 28: 'Linseed', 29: 'Maize', 30: 'Mango', 31: 'Masoor', 32: 'Mesta', 33: 'Moong(Green Gram)', 34: 'Niger seed', 35: 'Onion', 36: 'Orange', 37: 'Other  Rabi pulses', 38: 'Other Dry Fruit', 39: 'Other Fresh Fruits', 40: 'Other Kharif pulses', 41: 'Other Vegetables', 42: 'Papaya', 43: 'Peas  (vegetable)', 44: 'Pome Fruit', 45: 'Pome Granet', 46: 'Potato', 47: 'Ragi', 48: 'Rapeseed &Mustard', 49: 'Rice', 50: 'Safflower', 51: 'Samai', 52: 'Sapota', 53: 'Sesamum', 54: 'Small millets', 55: 'Soyabean', 56: 'Sugarcane', 57: 'Sunflower', 58: 'Sweet potato', 59: 'Tapioca', 60: 'Tobacco', 61: 'Tomato', 62: 'Turmeric', 63: 'Urad', 64: 'Varagu', 65: 'Wheat', 66: 'other fibres', 67: 'other misc. pulses', 68: 'other oilseeds'}


###################### APP ROutes #######################

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/fertilizer')
def fertilizer():
    return render_template("fertilizer.html")



@app.route('/crop')
def crop():
    return render_template("crop.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/ferti')
def ferti():
    return render_template("ferti.html")

@app.route('/ferti-predict', methods=['POST'])
def ferti_predict():
    title = 'Harvestify - Fertilizer Recommendation'

    if request.method == 'POST':
        Temperature = str(request.form['Temperature'])
        Humidity = int(request.form['nitrogen'])
        Moisture = int(request.form['Moisture'])
        SoilType = request.form['soiltype'] 
        CropType = request.form['cropname']
        Nitrogen = int(request.form['nitrogen'])
        Potassium = int(request.form['pottasium'])
        Phosphorous = int(request.form['phosphorous'])


        #Process soil type
        if SoilType == 'Black':
            SoilType = 0
        elif SoilType == 'Clayey':
            SoilType = 1
        elif SoilType == 'Loamy':
            SoilType = 2
        elif SoilType == 'Red':
            SoilType = 3
        elif SoilType == 'Sandy':
            SoilType = 4
        
        #Process crop type
        if CropType == 'Barley':
            CropType = 0
        elif CropType == 'Cotton':
            CropType = 1
        elif CropType == 'Ground Nuts':
            CropType = 2
        elif CropType == 'Maize':
            CropType = 3
        elif CropType == 'Millets':
            CropType = 4
        elif CropType == 'Oil seeds':
            CropType = 5
        elif CropType == 'Paddy':
            CropType = 6
        elif CropType == 'Pulses':
            CropType = 7
        elif CropType == 'Sugarcane':
            CropType = 8
        elif CropType == 'Tobacco':
            CropType = 9
        elif CropType == 'Wheat':
            CropType = 10


        data = np.array([[Temperature, Humidity, Moisture, SoilType, CropType, Nitrogen, Potassium, Phosphorous]])
        my_prediction = fert_recommendation_model.predict(data)
        final_prediction = my_prediction[0]

        #Process final prediction
        if final_prediction == 0:
            final_prediction = '10-26-26'
        elif final_prediction == 1:
            final_prediction = '14-35-14'
        elif final_prediction == 2:
            final_prediction = '17-17-17'
        elif final_prediction == 3:
            final_prediction = '20-20'
        elif final_prediction == 4:
            final_prediction = '28-28'
        elif final_prediction == 5:
            final_prediction = 'DAP'
        elif final_prediction == 6:
            final_prediction = 'Urea'
        

    return render_template('ferti-result.html', prediction=final_prediction, title=title)



@app.route('/yield', methods = ['POST', 'GET'])
def cropyield():
    title = 'Harvestify - Crop Yield Prediction'
    return render_template("yield.html")
# @app.route('/yield-predict', methods=['POST'])
# def yield_predict():
#     return render_template('yield-result.html')





@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])


    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]


    status_list = []

    if N > nr:
        status_list.append("NHigh")
    elif N < nr:
        status_list.append("Nlow")
    else:
        pass


    if P > pr:
        status_list.append("PHigh")
    elif P < pr:
        status_list.append("Plow")
    else:
        pass

    if K > kr:
        status_list.append("KHigh")
    elif K < kr:
        status_list.append("Klow")
    else:
        pass

    key = status_list

    response2 = None
    response3 = None

    if len(key) == 1:
        response1 = Markup(str(fertilizer_dic[key[0]]))
    
    elif len(key) == 0:
        response1 = Markup(str(fertilizer_dic['No']))

    elif len(key) == 2:
        response1 = Markup(str(fertilizer_dic[key[0]]))
        response2 = Markup(str(fertilizer_dic[key[1]]))
        

    elif len(key) == 3:
        response1 = Markup(str(fertilizer_dic[key[0]]))
        response2 = Markup(str(fertilizer_dic[key[1]]))
        response3 = Markup(str(fertilizer_dic[key[2]]))


    return render_template('fertilizer-result.html', recommendation=response1, recommendation2 = response2 , recommendation3 = response3, title=title)