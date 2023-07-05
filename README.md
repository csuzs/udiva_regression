# UDIVA Regression

implemented based on the following paper and data definition:

https://openaccess.thecvf.com/content/WACV2021W/HBU/papers/Palmero_Context-Aware_Personality_Inference_in_Dyadic_Scenarios_Introducing_the_UDIVA_Dataset_WACVW_2021_paper.pdf
https://data.chalearnlap.cvc.uab.es/UDIVA_ICCV2021/webpage/UDIVAv0.5_dataset_description.pdf

Current Model implementation doesn't get the extended context video chunks and metadata, 
but it is straightforward to implement based on current existing implementation.
It can get batches of chunks of videos.


Cutting off non-usable parts of videos based on defective_segments.json and task_limits.json
is not implemented, otherwise data pipeline is fully functional (to my current best knowledge)

Data gotten from Dataloader should be stacked into batches of chunks of videos. It is straightforward on image chunks, less so on audio chunks,
because feeding audio data into the audio network is tricky.

Images gotten from AVReader get_batch() method plotted with matplotlib.pyplot.imshow seems strange, feels like the colors are inverted

Image preprocess is still missing, it should be normalized and the strange image phenomenon fixed

The current state of main.py without the model initialization represents how far i've got,
feeding data into the model still needs some implementation (see points above), but training loop, evaluation and some kind of visualization is completely missing.

# Initializing the repository
Create a virtual environment and install packages in project root:
```shell
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```
annotation .zip files should be extracted in place in the directory of the database before running the code!
