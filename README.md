The backend of an app for reading and parsing the contents of a receipt image. `preprocess.py` reads an image file, robustly detects the receipt edges, finds the receipt corners, warps it to rectangular, scales it to a fixed size, and tweaks the contrast to improve OCR results. `read_receipt_image.py` accepts a preprocessed image, sends it to the [azure computer vision API](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/) for OCR and word-bounding box detection, parses the OCR output to figure out which words are items and which are prices, and converts the parsed contents to JSON. `app.py` wraps the functionality in a Flask app. 

This app is live at https://receipt-reader-bk.herokuapp.com/. You can send it an image with the Unix command `curl -F "file=<your image file>" https://receipt-reader-bk.herokuapp.com/`. The response is pretty slow, preumably because I'm on heroku's free tier.


## Testing

Start an app server locally with `heroku local`, and test it with `curl -F "file=<your image file>" http://0.0.0.0:5000/`. 


## Deployment

I basically followed heroku's instructions for deploying a flask app. `read_receipt_image.py` requires an API key to be saved in the environmental variable `AZURE_CV_KEY`. In order to get this into the heroku app (without storing it in git), use [config variables](https://devcenter.heroku.com/articles/config-vars). Once you're set up, you can push a new git commit directly to heroku with `git push heroku master`, and test it with `curl -F "file=<your image file>" https://<your_app_name>.herokuapp.com/``


## Status

I consider this to be about halfway to an beta version. Generally, it works well on a well-lit image taken by a sober person; however, for the use case I have in mind, it must be fairly robust to poor lighting conditions and camera shake.

The preprocessing is fairly robust, but the OCR leaves a lot to be desired. A final version would require a lot more error checking and spell checking. In some cases, the OCR may read a word as "Subto al" and the current version does not correct such obvious misspellings. It should include logic that enforces "subtotal + tax = total", "sum of all item prices=subtotal". etc. It should fail gracefully if it can't read an image. 

The scripts in this repo are a reasonable starting point for reading and parsing a photo of any sort of structured text document -- a check, a receipt, a form letter, etc.