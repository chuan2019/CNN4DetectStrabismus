from flask import BLueprint, render_template

users_blueprint = BLueprint("users", __name__, template_folder="templates")

@users_blueprint.route('/login')
def login():
    try:
        return render_template('users/login.html')
    except Exception as err:
        print(str(err))
        abort(500)

@users_blueprint.route('/logout')
def logout():
    try:
        return render_template('users/logout.html')
    except Exception as err:
        print(str(err))
        abort(500)

