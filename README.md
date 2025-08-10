<h2> Train model CNN + NLP with 5 Skin diseases dataset </h2>
<h3>Models</h3>
<p>ResNet50 + ClinicalBert</p>
<p>ResNet50 + BlueBert</p>
<p>DenseNet121 + ClinicalBert</p>
<p>DenseNet121 + BlueBert</p>
<p>EfficientNetv2-s + ClinicalBert</p>
<p>EfficientNetv2-s + BlueBert</p>
<h3>Step1: Data preparing </h3>
<p> Floder data  -> you can just load these floders and files for image (train,validate,test) for metadata (train.csv,val.csv,test.csv) </p>
<h3>Step2: Data preprocessing </h3>
<p> Floder preprocess_tensor -> this floder have 2 files of bert model which about text encoder model I have picked to test their abilities (BlueBert,ClinicalBert)</p>
<h3>Step3: Concatenation </h3>
<p> Floder concatfeature -> each file has CNN model (Efficientnetv2-s,Densenet121,ResNet50) and NLP model (BlueBert,ClinicalBert) concat with feature that be extract from each model and use concatenate technique before forward to training process</p>
<h3>Step4: Training </h3>
<p> Floder train -> start training if you have done 3 steps already you can run train file  after training you will have weight(.pth file best and last) you can create floder weights to keep it </p>
<h3>Step5: Evaluation </h3>
<p> Floder evaluate -> after you had created weights floder you can bring weight to evaluate to see model performance if you run evaluate file, The result will have (confusion matrixs,correct and incorrect images, performances)</p>
<h3>Step6: Apply </h3>
<p> File app -> run this file to use model on localhost web</p>
