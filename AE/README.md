## ProvIot

### Instructions to run [ProvIoT](provIoT.py)
- Driver script for ProvIoT, which is an AE based IDS that detects anomalous paths using graph attributes, e.g., node/edge labels.
```commandline
python provIoT.py
```

Output:

```commandline
...
2024-03-04 17:37:08,387 | INFO	| ************************** Evalaution Result **************************
2024-03-04 17:37:08,387 | INFO	| Threshold= 0.007674000090030207
2024-03-04 17:37:08,388 | INFO	| TP:262
2024-03-04 17:37:08,388 | INFO	| FP:46
2024-03-04 17:37:08,388 | INFO	| TN:220
2024-03-04 17:37:08,389 | INFO	| FN:4
2024-03-04 17:37:08,389 | INFO	| Accuracy:0.9060150375939849
2024-03-04 17:37:08,389 | INFO	| Precision:0.8506493506493507
2024-03-04 17:37:08,389 | INFO	| Recall:0.9849624060150376
2024-03-04 17:37:08,389 | INFO	| F1:0.9128919860627178
2024-03-04 17:37:08,443 | INFO	| Threshold = 0.007674000090030207
```