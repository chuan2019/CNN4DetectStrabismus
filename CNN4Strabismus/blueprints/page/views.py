from flask import Blueprint, render_template

page = Blueprint('page', __name__, template_folder='templates')

@page.route('/')
def home():
    return render_template('page/home.html')

@page.route('/terms')
def terms():
    return render_template('page/terms.html')

@page.route('/privacy')
def privacy():
    return render_template('page/privacy.html')

@page.route('/capture')
def capture_picture():
    return render_template('page/capture_picture.html')

@page.route('/upload')
def upload_picture():
    return render_template('page/upload_picture.html')
