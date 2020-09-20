from flask import Blueprint, render_template

photo_blueprint = Blueprint('photo', __name__,
                                     template_folder='templates')

@photo_blueprint.route('/photo')
def photo():
    try:
        return render_template('photo/photo.html')
    except Exception as err:
        print(str(err))
        abort(404)

