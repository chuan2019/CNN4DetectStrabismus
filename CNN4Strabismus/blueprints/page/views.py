from flask import Blueprint, render_template

page = Blueprint('page', __name__,
                         template_folder='templates',
                         static_folder='static')

@page.route('/')
def home():
    print(f'page.root_path = {page.root_path}')
    try:
        return render_template('page/home.html')
    except TemplateNotFound:
        abort(404)

@page.route('/terms')
def terms():
    try:
        return render_template('page/terms.html')
    except TemlateNotFound:
        abort(404)

@page.route('/privacy')
def privacy():
    try:
        return render_template('page/privacy.html')
    except TemplateNotFound:
        abort(404)

@page.route('/capture')
def capture_picture():
    try:
        return render_template('page/capture_picture.html')
    except TemplateNotFound:
        abort(404)

@page.route('/upload')
def upload_picture():
    try:
        return render_template('page/upload_picture.html')
    except TemplateNotFound:
        abort(404)

