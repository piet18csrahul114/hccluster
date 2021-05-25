%%writefile app.py
import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('/content/drive/My Drive/hccluster.pkl','rb'))   
dataset= pd.read_csv('/content/drive/My Drive/Wholesale customers data.csv')
X = dataset.iloc[:,2:8].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_note_authentication(channel1,channel2,channel3,channel4,channel5,region1,region2,region3,region4,region5,fresh1,milk1,grocery1,frozen1,detergents1,delicassen1,fresh2,milk2,grocery2,frozen2,detergents2,delicassen2,fresh3,milk3,grocery3,frozen3,detergents3,delicassen3,fresh4,milk4,grocery4,frozen4,detergents4,delicassen4,fresh5,milk5,grocery5,frozen5,detergents5,delicassen5):
  predict= model.fit_predict(sc.transform([[fresh1,milk1,grocery1,frozen1,detergents1,delicassen1 ],[fresh2,milk2,grocery2,frozen2,detergents2,delicassen2 ],[fresh3,milk3,grocery3,frozen3,detergents3,delicassen3 ],[fresh4,milk4,grocery4,frozen4,detergents4,delicassen4 ],[fresh5,milk5,grocery5,frozen5,detergents5,delicassen5 ]]))
  print(predict)
  res = []
  for i in predict:
    if i==[0]:
      res.append("Customer is careless")
    elif i==[1]:
      res.append("Customer is standard")
    elif i==[2]:
      res.append("Customer is target")
    elif i==[3]:
      res.append("Customer is careful")
    else:
      res.append("Customer is sensible")
  return res
  
def main():
    
      html_temp = """
    <div class="" style="background-color:yellow;" >
    <div class="clearfix">           
    <div class="col-md-12">
    <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
    <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
    <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning Lab Experiment</p></center> 
    </div>
    </div>
    </div>
    """
      st.markdown(html_temp,unsafe_allow_html=True)
      st.header("Customer Segmentation on wholesale data ")
      
      channel1 = st.selectbox(
      "Channel1",
      ("1", "2")
      )
      region1 = st.selectbox(
      "Region1",
      ("1", "2","3")
      )
      fresh1 = st.number_input('Insert fresh1',3,112151)
      milk1 = st.number_input('Insert milk1',55,73498)
      grocery1 = st.number_input('Insert grocery1',3,92780)
      frozen1 = st.number_input('Insert frozen1',25,60869)
      detergents1 = st.number_input('Insert Detergents_Paper1',3,40827)
      delicassen1 = st.number_input('Insert Delicassen1',3,47943)

      channel2 = st.selectbox(
      "Channel2",
      ("1", "2")
      )
      region2 = st.selectbox(
      "Region2",
      ("1", "2","3")
      )
      fresh2 = st.number_input('Insert fresh2',3,112151)
      milk2 = st.number_input('Insert milk2',55,73498)
      grocery2 = st.number_input('Insert grocery2',3,92780)
      frozen2 = st.number_input('Insert frozen2',25,60869)
      detergents2 = st.number_input('Insert Detergents_Paper2',3,40827)
      delicassen2 = st.number_input('Insert Delicassen2',3,47943)

      channel3 = st.selectbox(
      "Channel3",
      ("1", "2")
      )
      region3 = st.selectbox(
      "Region3",
      ("1", "2","3")
      )
      fresh3 = st.number_input('Insert fresh3',3,112151)
      milk3 = st.number_input('Insert milk3',55,73498)
      grocery3 = st.number_input('Insert grocery3',3,92780)
      frozen3 = st.number_input('Insert frozen3',25,60869)
      detergents3 = st.number_input('Insert Detergents_Paper3',3,40827)
      delicassen3 = st.number_input('Insert Delicassen3',3,47943)

      channel4 = st.selectbox(
      "Channel4",
      ("1", "2")
      )
      region4 = st.selectbox(
      "Region4",
      ("1", "2","3")
      )
      fresh4 = st.number_input('Insert fresh4',3,112151)
      milk4 = st.number_input('Insert milk4',55,73498)
      grocery4 = st.number_input('Insert grocery4',3,92780)
      frozen4 = st.number_input('Insert frozen4',25,60869)
      detergents4 = st.number_input('Insert Detergents_Paper4',3,40827)
      delicassen4 = st.number_input('Insert Delicassen4',3,47943)

      channel5 = st.selectbox(
      "Channel5",
      ("1", "2")
      )
      region5 = st.selectbox(
      "Region5",
      ("1", "2","3")
      )
      fresh5 = st.number_input('Insert fresh5',3,112151)
      milk5 = st.number_input('Insert milk5',55,73498)
      grocery5 = st.number_input('Insert grocery5',3,92780)
      frozen5 = st.number_input('Insert frozen5',25,60869)
      detergents5 = st.number_input('Insert Detergents_Paper5',3,40827)
      delicassen5 = st.number_input('Insert Delicassen5',3,47943)

    
      result = []
      if st.button("Predict"):
        result=predict_note_authentication(channel1,channel2,channel3,channel4,channel5,region1,region2,region3,region4,region5,fresh1,milk1,grocery1,frozen1,detergents1,delicassen1,fresh2,milk2,grocery2,frozen2,detergents2,delicassen2,fresh3,milk3,grocery3,frozen3,detergents3,delicassen3,fresh4,milk4,grocery4,frozen4,detergents4,delicassen4,fresh5,milk5,grocery5,frozen5,detergents5,delicassen5)
        st.success('Model has predicted {}'.format(result))

      if st.button("About"):
        st.subheader("Developed by Rahul Chhablani")
        st.subheader("Department of Computer Engineering") 

if __name__=='__main__':
  main()