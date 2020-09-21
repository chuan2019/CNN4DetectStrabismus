from flask import Blueprint, render_template

video_blueprint = Blueprint('video', __name__, template_folder='templates')

@video_blueprint.route('/video')
def video_home():
    try:
        return render_template('video/video.html')
    except Exception as err:
        print(str(err))
        abort(404)



