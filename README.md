<h2> Train model CNN + NLP with 5 Skin diseases dataset</h2>
<h3>Step1: Data preparing </h3>
<p> Floder data  -> you can just load these floders and files for image (train,validate,test) for metadata (train.csv,val.csv,test.csv) </p>
<h3>Step2: Data preprocessing </h3>
<p> Floder preprocess_tensor -> this floder have 2 files of bert model which about text encoder model I have picked to test their abilities (BlueBert,ClinicalBert)</p>
<h3>Step3: Concatenation </h3>
<p> Floder concatfeature -> each file has CNN model (Efficientnetv2-s,Densenet121,ResNet50) and NLP model (BlueBert,ClinicalBert) concat with feature that be extract from each model and use concatenate technique before forward to training process</p>
<h3>Step4: Training </h3>
