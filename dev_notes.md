# Metrics
- For baseline don't do any augmentation
	- can specify `weighted_...` in `.compile` to introduce weighted accuracy/f1 score
- undersampling ok
- oversampling possible future approach
- Augmenting might generate too much noise.

## Weighted accuracy and f1 scores
Okii so looks like we have to do it something like this:
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', weighted_metrics=['f1_score', 'accuracy'], ...)
```


