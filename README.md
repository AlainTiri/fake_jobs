# fake_jobs

## Use case

Model and create an API point for an AI that detects fake job ads in the US market.

The dataset:
- A dataset comprising text descriptions of jobs and their true or false job status (0 or 1).
- It is clearly unbalanced, because only 5% of the descriptions are labeled "false jobs".
- There are many duplicates, which requires data engineering.

Equation to solve:
- There must be a balance between the short-term objective, the detection of false announcements, and the future improvement of the model.

Proposed solution:
Work on the accumulation of new validated data to allow a significant rebalancing of the dataset for false advertisements and not to block truthful advertisers in the event of publication on a website.
The proposed model detects around 99% of real ads (recall score), but only 56% of fake ones. This means that for 318 ads received, the model will return 73 for post processing, of which +- only 1 will be a real ad. (note: it is possible to continue looking for another ratio, the code present in the notebook "with nltk" allows it easily)
At this level, it is possible to either hire someone to detect which is the right announcement, or to keep everything for the next model training.
Everything works via FastAPI in a Docker.

The technology stack:
- Python + Jupyter Notebook
- Scikit-learn
-FastAPI

                Reshape
                0    0.611461
                1    0.388539
                Name: fraudulent, dtype: float64
                              precision    recall  f1-score   support
                
                           0       0.77      0.99      0.87       189
                           1       0.99      0.56      0.71       129
                
                    accuracy                           0.82       318
                   macro avg       0.88      0.78      0.79       318
                weighted avg       0.86      0.82      0.80       318
                
                col_0         0   1
                fraudulent         
                0           188   1
                1            57  72
