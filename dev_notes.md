# Baseline
1/14 = 0.07 - lower limit accuracy of random assignment

tl;dr VGG16 was huge and overfitted... so we need a smaller model

...
For the baseline we opted for a structure like:
```
 Layer (type)                Output Shape              Param #   
=================================================================
 vgg16 (Functional)          (None, 7, 7, 512)         14714688  
                                                                 
 flatten_11 (Flatten)        (None, 25088)             0         
                                                                 
 dense_22 (Dense)            (None, 50)                1254450   
                                                                 
 dense_23 (Dense)            (None, 14)                714       
                                                                 
=================================================================
Total params: 15,969,852
Trainable params: 1,255,164
Non-trainable params: 14,714,688
```

## Results and Analysis
Here our loss and accuracy curves looked like this:
![image](https://github.com/nikkernoodle/shark-id/assets/36482217/79b5a20b-0b65-4f30-9cba-9e8605cb8799)
we subsequently decided this model sucked (technical term) and would try a new model in order to avoid this dramatic overfitting...

Investigation of the confusion matrix yielded further insights into the nature of our problem. 
![image](https://github.com/nikkernoodle/shark-id/assets/36482217/7265fc94-f33e-4cb3-bc84-f94b4ac18646)
It's possible to see here the some categories (e.g. whale, thresher and basking) are being identified quite well whereas most of the rest are not being classified well but mostly identifying the images.
Looking into weighted and macro metrics we can see that the balance of our dataset seems to not be affecting the results so much.
|metric|macro(/regular accuracy)|weighted(/balanced accuracy)|
|----|----|----|
|accuracy|43.023%|42.999%|
|precision|42.129%|44.772%|
|recall|42.999%|43.023%|
|f1|41.245%|42.764%|

**therefore, the likely cause is overfitting**

# Metrics
- For baseline don't do any augmentation
	- can specify `weighted_...` in `.compile` to introduce weighted accuracy/f1 score
- undersampling ok
- oversampling possible future approach
- Augmenting might generate too much noise.

## Evaluation Metrics as a Post-Processing Step
macro and weighted versions of
- f1 $$\frac{2(Precision \times Recall)}{Precision + Recall} = \frac{2(\frac{TP}{TP + FP}\frac{TP}{TP + FN})}{\frac{TP}{TP + FP} + \frac{TP}{TP + FN}}$$
- precision $$\frac{TP}{TP + FP}$$
- recall $$\frac{TP}{TP + FN}$$
- accuracy $$\frac{TP + TN}{TP + TN + FP + FN}$$
_Here, $TP$ refers to "True Positive" etc._

each score was implemented from `sklearn` in an evaluation step.

## Weighted accuracy and f1 scores (only available through `tf-nightly`)
Okii so looks like we have to do it something like this:
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', weighted_metrics=['f1_score', 'accuracy'], ...)
```
