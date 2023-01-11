# My drafts for modeling

## Transfert learning with Bert
I started with a transfer learning initiative based on the pattern of sentiment analysis, to see what it could do. The modeling ran on a Google Collab, as I don't have a GPU on my current machine. The result was unsatisfactory, random, etc. And above all, the training time was long.

## ML with nltk
I switched to normal ML on a Notebook. I tested many supervised clustering models and SVM seemed to me to be more suitable with this unbalanced dataset.

The notebook shows the result of several initiatives, some of which were abandoned to manage the delivery time of the project. The training time for complex models, such as Ensemble Learning or XGBoost, was penalizing. With more time, it is likely to be able to test other solutions, but the balance found with SVM seems good to me as of now.

The selected code, which will be better explained in the notebook in the next few hours, is the result of numerous tests, in particular on the size of the dataset. Indeed, to avoid imbalances, I practiced a reduction in the size of the training dataset of the real announcements. The compromise found is to have twice as many real ads as fake ones.

