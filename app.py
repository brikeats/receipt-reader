"""
Flask Documentation:     http://flask.pocoo.org/docs/
Jinja2 Documentation:    http://jinja.pocoo.org/2/documentation/
Werkzeug Documentation:  http://werkzeug.pocoo.org/documentation/

This file creates your application.
"""

import os
from flask import Flask, render_template, jsonify

app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'this_should_be_configured')


@app.route('/')
def home():
    """Render website's home page."""
    # return render_template('home.html')

    items = ['sierra nevada stout', 'lagunitas ipa', 'founders centennial']
    prices = [5.50, 6.50, 7.50]
    subtotal = sum(prices)
    tax = subtotal * 0.07
    total = subtotal + tax
    bill_dict = {
        'subtotal': subtotal,
        'tax': tax,
        'total': total
    }
    return jsonify(bill_dict)


@app.errorhandler(404)
def page_not_found(error):
    """Custom 404 page."""
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True)
