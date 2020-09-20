from flask import Blueprint, render_template

page_blueprint = Blueprint('page', __name__,
                                   template_folder='templates',
                                   static_folder='static')

@page_blueprint.route('/')
def home():
    #print(f'page_blueprint.root_path = {page_blueprint.root_path}')
    try:
        return render_template('page/index.html')
    except TemplateNotFound:
        abort(404)

@page_blueprint.route('/terms')
def terms():
    try:
        return render_template('page/terms.html')
    except TemlateNotFound:
        abort(404)

@page_blueprint.route('/privacy')
def privacy():
    try:
        return render_template('page/privacy.html')
    except TemplateNotFound:
        abort(404)

